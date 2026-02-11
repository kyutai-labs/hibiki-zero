# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import math
import random
import time
from dataclasses import dataclass

import aiohttp
import numpy as np
import sentencepiece
import sphn
import torch
from aiohttp import web
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.run_inference import get_condition_tensors

from hibiki_zero.client_utils import log


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(
        self,
        model_type: str,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        device: str | torch.device,
        **kwargs,
    ):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=1)
        self.lm_gen = LMGen(lm, cfg_coef=1, condition_tensors=condition_tensors, **kwargs)

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])

        torch.cuda.synchronize()

    async def decode_and_send(
        self, tokens: torch.Tensor, ws: web.WebSocketResponse, opus_writer: sphn.OpusStreamWriter
    ):
        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
        main_pcm = self.mimi.decode(tokens[:, 1:])
        main_pcm = main_pcm.cpu()
        opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
        if len(opus_bytes) > 0:
            await ws.send_bytes(b"\x01" + opus_bytes)
        text_token = tokens[0, 0, 0].item()
        if text_token not in (0, 3):
            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
            _text = _text.replace("▁", " ")
            msg = b"\x02" + bytes(_text, encoding="utf8")
            log("info", f"text token: '{_text}'")
            await ws.send_bytes(msg)
        elif text_token == 2:
            log("info", "End Of Sequence token")

    async def recv_loop(
        self,
        ws: web.WebSocketResponse,
        opus_reader: sphn.OpusStreamReader,
        opus_writer: sphn.OpusStreamWriter,
    ):
        all_pcm_data = None
        skip_frames = 1
        try:
            frame_idx: int = 0
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    log("error", f"{ws.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    log("error", f"Unexpected message type {message.type}")
                    continue
                message = message.data
                if not isinstance(message, bytes):
                    log("error", f"Unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    log("warning", "Empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    pcm = opus_reader.append_bytes(payload)
                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        be = time.time()
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size :]
                        chunk = torch.from_numpy(chunk)
                        chunk = chunk.to(device=self.device)[None, None]
                        codes = self.mimi.encode(chunk)
                        if skip_frames:
                            # The first input audio frame is ignored, as from the point of
                            # view of the model it is in the past. We still `mimi.encode` for simplicity,
                            # however as the first encoded frame has a specific structure (due to the left padding),
                            # we reset the streaming state of the encoder to reapply the padding on the next call.
                            self.mimi.reset_streaming()
                            skip_frames -= 1
                        for c in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                            if tokens is None:
                                continue
                            await self.decode_and_send(tokens, ws, opus_writer)
                        log(
                            "info",
                            f"frame {frame_idx} handled in {1000 * (time.time() - be):.1f}ms",
                        )
                        frame_idx += 1
                else:
                    log("warning", f"unknown message kind {kind}")
        finally:
            log("info", "Connection closed.")

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        log("info", "Accepted connection.")

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await self.recv_loop(ws, opus_reader, opus_writer)
        log("info", "Done with connection.")
        return ws


def get_lmgen(
    lm: LMModel, checkpoint_info: loaders.CheckpointInfo, batch_size: int, cfg_coef: int = 1.0
) -> LMGen:
    condition_tensors = get_condition_tensors(
        checkpoint_info.model_type, lm, batch_size=batch_size, cfg_coef=cfg_coef
    )
    lm_gen = LMGen(
        lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **checkpoint_info.lm_gen_config
    )
    return lm_gen


def add_input_eos(
    codes: torch.Tensor, mimi: MimiModel, audio_durations: list[float]
) -> torch.Tensor:
    other_audio_eos_idx: torch.Tensor = torch.tensor(
        [
            min(math.ceil(duration * mimi.frame_rate), codes.shape[-1] - 1)
            for duration in audio_durations
        ]
    )[:, None, None].to(codes.device)  # B, 1, 1
    codes_like_indexes: torch.Tensor = torch.arange(0, codes.shape[-1])[None, None].to(
        codes.device
    )  # 1, 1, T
    codes_with_input_eos: torch.Tensor = torch.where(
        codes_like_indexes == other_audio_eos_idx,
        torch.full([1], mimi.cardinality, device=codes.device),
        codes,
    )  # B, K, T
    return codes_with_input_eos


def encode_inputs(
    batch_wavs: torch.Tensor, mimi: MimiModel, lm_gen: LMGen, audio_durations: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    with torch.no_grad():
        codes: torch.Tensor = mimi.encode(batch_wavs.to(lm_gen.lm_model.device))
        codes_with_input_eos = add_input_eos(codes, mimi, audio_durations)
        warmup_wav: torch.Tensor = torch.zeros(
            codes.shape[0],
            1,
            frame_size * lm_gen.max_delay,
            dtype=torch.float32,
            device=codes.device,
        )
        warmup_codes: torch.Tensor = mimi.encode(warmup_wav)
    return codes_with_input_eos, warmup_codes


def decode_outputs(
    batch_codes: torch.Tensor,
    batch_text_tokens: torch.Tensor,
    mimi: MimiModel,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
) -> list[tuple[torch.Tensor, str]]:
    with torch.no_grad():
        output_wavs: torch.Tensor = mimi.decode(batch_codes).cpu()

    outputs: list[tuple[torch.Tensor, str]] = []
    for output_idx, wav in enumerate(output_wavs):
        text_tokens: list[int] = batch_text_tokens[output_idx].tolist()
        if text_tokenizer.eos_id() in text_tokens:
            eos_idx: int = text_tokens.index(text_tokenizer.eos_id())
        else:
            log(
                "warning",
                f"The model didn't generate output EOS token for entry {output_idx}, truncating audio after the last word generated.",
            )
            eos_idx: int = len(text_tokens) - 1
            while eos_idx > 0 and text_tokens[eos_idx] == text_tokenizer.pad_id():
                eos_idx -= 1
        text_tokens = [t for t in text_tokens[:eos_idx] if t > text_tokenizer.pad_id()]
        text: str = text_tokenizer.decode(text_tokens)
        wav = wav[:, : int(eos_idx * mimi.sample_rate / mimi.frame_rate)]
        outputs.append((wav, text))

    return outputs
