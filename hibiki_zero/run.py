# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os
import secrets
import tarfile
import time
from pathlib import Path
from typing import Optional

import torch
import typer
from aiohttp import web
from huggingface_hub import hf_hub_download
from moshi.models import LMGen, loaders
from typing_extensions import Annotated

from hibiki_zero.client_utils import audio_read, log, save_results, stack_and_pad_audio
from hibiki_zero.inference import ServerState, decode_outputs, encode_inputs, get_lmgen, seed_all

ROOT_DIR: Path = Path(__file__).parent.parent
DEFAULT_REPO: str = "kyutai/hibiki-zero-3b-pytorch-bf16"
DEFAULT_AUDIO_SAMPLES: list[Path] = [
    ROOT_DIR / "samples" / fname for fname in os.listdir(ROOT_DIR / "samples")
]

cli_app = typer.Typer()


@cli_app.command()
@torch.no_grad()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind the server to.")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind the server to.")] = 8998,
    static: Annotated[Optional[str], typer.Option(help="Path to static files directory.")] = None,
    gradio_tunnel: Annotated[bool, typer.Option(help="Activate a gradio tunnel.")] = False,
    gradio_tunnel_token: Annotated[Optional[str], typer.Option(help="Custom tunnel token.")] = None,
    hf_repo: Annotated[
        str, typer.Option(help="HF repo for model, codec and text tokenizer.")
    ] = DEFAULT_REPO,
    config_path: Annotated[Optional[str], typer.Option(help="Path to a config file.")] = None,
    tokenizer: Annotated[Optional[str], typer.Option(help="Path to a text tokenizer file.")] = None,
    model_weight: Annotated[
        Optional[str], typer.Option(help="Path to a Hibiki-Zero checkpoint.")
    ] = None,
    mimi_weight: Annotated[
        Optional[str], typer.Option(help="Path to a Mimi codec checkpoint.")
    ] = None,
    lora_weight: Annotated[Optional[str], typer.Option(help="Path to a LoRA checkpoint.")] = None,
    fuse_lora: Annotated[
        bool, typer.Option("--fuse-lora/--no-fuse-lora", help="Fuse LoRA layers.")
    ] = True,
    bf16: Annotated[bool, typer.Option(help="Use bfloat16.")] = False,
    device: Annotated[str, typer.Option(help="Device to run on.")] = "cuda",
    ssl: Annotated[
        Optional[str], typer.Option(help="Directory containing cert.pem and key.pem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
):
    # sanity checks
    if not torch.cuda.is_available():
        log(
            "error",
            "Found no NVIDIA driver on your system. The server needs to be launched from a machine that has access to a GPU.",
        )
        return

    seed_all(seed)
    dtype = torch.bfloat16 if bf16 else torch.float16

    log("info", "Starting Hibiki-Zero server.")
    setup_tunnel, tunnel_token = None, ""
    if gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log(
                "error", "Gradio is required for tunnel support. Install with `pip install gradio`."
            )
            raise typer.Exit(1)
        setup_tunnel = networking.setup_tunnel
        tunnel_token = (
            secrets.token_urlsafe(32) if gradio_tunnel_token is None else gradio_tunnel_token
        )

    log("info", "Retrieving the model checkpoint...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo,
        model_weight,
        mimi_weight,
        tokenizer,
        lora_weights=lora_weight,
        config_path=config_path,
    )

    log("info", "Loading the codec {0}", [(checkpoint_info.mimi_weights, "blue")])
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "Loading the model {0}", [(checkpoint_info.moshi_weights, "blue")])
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype, fuse_lora=fuse_lora)

    state = ServerState(
        checkpoint_info.model_type,
        mimi,
        text_tokenizer,
        lm,
        device,
        **checkpoint_info.lm_gen_config,
    )
    log("info", "Warming up the model...")
    state.warmup()

    web_app = web.Application()
    web_app.router.add_get("/api/chat", state.handle_chat)

    static_path: Optional[str] = None
    if static is None:
        log("info", "Retrieving the static content...")
        dist_tgz = Path(hf_hub_download("kyutai/moshi-artifacts", "dist.tgz"))
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
    elif static != "none":
        static_path = static

    if static_path is not None:

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        web_app.router.add_get("/", handle_root)
        web_app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    protocol, ssl_context = "http", None
    if ssl is not None:
        import ssl as ssl_module

        ssl_context = ssl_module.create_default_context(ssl_module.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(
            certfile=os.path.join(ssl, "cert.pem"), keyfile=os.path.join(ssl, "key.pem")
        )
        protocol = "https"

    local_url = f"{protocol}://{host}:{port}"
    log("info", "Access the Web UI directly at {0}", [(local_url, "orange")])

    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel("localhost", port, tunnel_token, None, **tunnel_kwargs)  # type: ignore
        log("info", "Tunnel started at {0}", [(tunnel, "green")])
        log("info", "Note: tunnel goes through the US; expect higher latency in Europe.")

    web.run_app(web_app, host=host, port=port, ssl_context=ssl_context)


@cli_app.command()
@torch.no_grad()
def generate(
    files: Annotated[list[Path], typer.Option("--file", help="Input files to translate.")] = None,
    gen_duration: Annotated[
        float,
        typer.Option(
            help="Generation duration in seconds. Should be <=120 seconds for Hibiki-Zero."
        ),
    ] = 120,
    out_dir: Annotated[Path, typer.Option(help="Directory where to save the outputs.")] = None,
    tag: Annotated[
        str, typer.Option(help="Tag to add to translation outputs filenames to identify them.")
    ] = None,
    repeats: Annotated[int, typer.Option(help="Do repeats generation for each input file.")] = 1,
    hf_repo: Annotated[
        str, typer.Option(help="HF repo for model, codec and text tokenizer.")
    ] = DEFAULT_REPO,
    config_path: Annotated[Optional[str], typer.Option(help="Path to a config file.")] = None,
    tokenizer: Annotated[Optional[str], typer.Option(help="Path to a text tokenizer file.")] = None,
    model_weight: Annotated[
        Optional[str], typer.Option(help="Path to a Hibiki-Zero checkpoint.")
    ] = None,
    mimi_weight: Annotated[
        Optional[str], typer.Option(help="Path to a Mimi codec checkpoint.")
    ] = None,
    lora_weight: Annotated[Optional[str], typer.Option(help="Path to a LoRA checkpoint.")] = None,
    fuse_lora: Annotated[
        bool, typer.Option("--fuse-lora/--no-fuse-lora", help="Fuse LoRA layers.")
    ] = True,
    bf16: Annotated[bool, typer.Option(help="Use bfloat16.")] = False,
    device: Annotated[str, typer.Option(help="Device to run on.")] = "cuda",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
):
    if not torch.cuda.is_available():
        log(
            "error",
            "Found no NVIDIA driver on your system. Generation needs to be done on a machine that has access to a GPU.",
        )
        return

    seed_all(seed)
    dtype = torch.bfloat16 if bf16 else torch.float16

    log("info", "Starting Hibiki-Zero inference.")
    files = files if files is not None else DEFAULT_AUDIO_SAMPLES
    files = [fpath for fpath in files for _ in range(repeats)]
    all_files_exist: bool = len(files) > 0
    for fpath in files:
        if not fpath.exists():
            log("error", f"File not found: {fpath}")
            all_files_exist = False
    if not all_files_exist:
        if len(files) == 0:
            log("error", "No files provided.")
        return
    log("info", "The following audios will be processed in a single batch:")
    for fidx, fpath in enumerate(files):
        log("info", f"{fidx} : " + "{0}", [(fpath, "grey")])

    log("info", "Retrieving the model checkpoint...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo,
        model_weight,
        mimi_weight,
        tokenizer,
        lora_weights=lora_weight,
        config_path=config_path,
    )

    log("info", "Loading the codec {0}", [(checkpoint_info.mimi_weights, "blue")])
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "Loading the model {0}", [(checkpoint_info.moshi_weights, "blue")])
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype, fuse_lora=fuse_lora)

    log("info", "Loading audios...")
    input_wavs: list[torch.Tensor] = [
        audio_read(fpath, to_sample_rate=mimi.sample_rate, mono=True)[0] for fpath in files
    ]
    audio_durations: list[float] = [wav.shape[-1] / mimi.sample_rate for wav in input_wavs]
    if max(audio_durations) > gen_duration:
        log(
            "error",
            f"One of the input audios is longer than the gen duration: {max(audio_durations)} > {gen_duration=}",
        )
        return
    batch_wavs = stack_and_pad_audio(input_wavs, max_len=int(gen_duration * mimi.sample_rate))
    batch_size: int = batch_wavs.shape[0]

    lm_gen: LMGen = get_lmgen(lm, checkpoint_info, batch_size)

    log("info", "Encoding audios...")
    codes, warmup_codes = encode_inputs(batch_wavs, mimi, lm_gen, audio_durations)

    output_text_tokens: list[torch.Tensor] = []
    output_audio_tokens: list[torch.Tensor] = []
    gen_steps: int = codes.shape[-1]
    start_gen_time: float = time.time()
    with torch.no_grad(), lm_gen.streaming(batch_size):
        # warmup
        for step in range(warmup_codes.shape[-1]):
            _ = lm_gen.step(warmup_codes[:, :, step : step + 1])
        # generation
        for step in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, step : step + 1])
            if tokens is None:
                print(None)
            else:
                output_text_tokens.append(tokens[:, 0, :])
                output_audio_tokens.append(tokens[:, 1:, :])
            log(
                "info",
                f"Running inference: {step}/{gen_steps} steps = {step / gen_steps:.0%}",
                end="\r",
            )
    gen_time: float = time.time() - start_gen_time
    real_time_factor: float = batch_wavs.shape[-1] / mimi.sample_rate / gen_time
    throughput: float = real_time_factor * batch_size
    log(
        "info",
        "Generated outputs in {0} "
        + f"(throughput = batch size x real-time factor = {throughput:.1f})",
        [(f"{real_time_factor:.1f}x real-time", "orange")],
    )

    log("info", "Saving results...")
    batch_text_tokens: torch.Tensor = torch.concat(output_text_tokens, dim=-1)  # B x T
    batch_codes: torch.Tensor = torch.concat(output_audio_tokens, dim=-1)  # B x K x T
    outputs = decode_outputs(batch_codes, batch_text_tokens, mimi, text_tokenizer)
    output_dir: Path = out_dir if out_dir is not None else ROOT_DIR / "translations"
    save_results(
        inputs=zip(files, input_wavs),
        outputs=outputs,
        sample_rate=mimi.sample_rate,
        output_dir=output_dir,
        tag=tag,
    )
    log("info", "Saved translation results in {0}", [(output_dir, "green")])


def main():
    cli_app()


if __name__ == "__main__":
    main()
