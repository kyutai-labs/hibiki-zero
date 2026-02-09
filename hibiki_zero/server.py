# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from pathlib import Path
import tarfile
import secrets
import argparse
import inspect
from aiohttp import web
import torch
from huggingface_hub import hf_hub_download

from moshi.client_utils import log
from moshi.models import loaders
from hibiki_zero.state import seed_all, ServerState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action="store_true", help="Activate a gradio tunnel.")
    parser.add_argument(
        "--gradio-tunnel-token",
        help="Provide a custom (secret) token here to keep getting the same URL.",
    )
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi."
    )
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into to load the model, codec and text tokenizer.",
    )
    parser.add_argument(
        "--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to a local config file.", default=None
    )
    parser.add_argument("--cfg-coef", type=float, default=1.0, help="CFG coefficient.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'."
    )
    parser.add_argument(
        "--no_fuse_lora",
        action="store_false",
        dest="fuse_lora",
        default=True,
        help="Do not fuse LoRA layers intot Linear layers.",
    )
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16, not bfloat16, better for old GPUs.",
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )

    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log(
                "error",
                "Cannot find gradio which is required to activate a tunnel. "
                "Please install with `pip install gradio`.",
            )
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    log("info", "Starting Hibiki-Zero server.")

    log("info", "Retrieving the model checkpoint...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        args.moshi_weight,
        args.mimi_weight,
        args.tokenizer,
        lora_weights=args.lora_weight,
        config_path=args.config_path,
    )
    log("info", "Loading the codec...")
    mimi = checkpoint_info.get_mimi(device=args.device)
    # log("info", "Mimi loaded!")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "Loading the model...")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora)
    # log("info", "Hibiki-Zero loaded!")

    state = ServerState(
        checkpoint_info.model_type,
        mimi,
        text_tokenizer,
        lm,
        args.cfg_coef,
        args.device,
        **checkpoint_info.lm_gen_config,
    )
    log("info", "Warming up the model...")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    static_path: None | str = None
    if args.static is None:
        log("info", "Retrieving the static content...")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        # When set to the "none" string, we don't serve any static content.
        static_path = args.static
    if static_path is not None:

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"Serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None, **tunnel_kwargs)  # type: ignore
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}")
        log(
            "info",
            "Note that this tunnel goes through the US and you might experience high latency in Europe.",
        )
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
