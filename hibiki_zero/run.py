# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import inspect
import tarfile
import secrets
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import torch
import typer
from aiohttp import web
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from hibiki_zero.client_utils import log
from hibiki_zero.state import seed_all, ServerState

DEFAULT_REPO: str = "kyutai/hibiki-zero-3b-pytorch-bf16"

cli_app = typer.Typer()


@cli_app.command()
@torch.no_grad()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind the server to.")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind the server to.")] = 8998,
    static: Annotated[
        Optional[str], typer.Option(help="Path to static files directory, or 'none'.")
    ] = None,
    gradio_tunnel: Annotated[bool, typer.Option(help="Activate a gradio tunnel.")] = False,
    gradio_tunnel_token: Annotated[Optional[str], typer.Option(help="Custom tunnel token.")] = None,
    tokenizer: Annotated[Optional[str], typer.Option(help="Path to a text tokenizer file.")] = None,
    moshi_weight: Annotated[
        Optional[str], typer.Option(help="Path to a Hibiki-Zero checkpoint.")
    ] = None,
    mimi_weight: Annotated[Optional[str], typer.Option(help="Path to a Mimi checkpoint.")] = None,
    hf_repo: Annotated[
        str, typer.Option(help="HF repo for model, codec and text tokenizer.")
    ] = DEFAULT_REPO,
    lora_weight: Annotated[Optional[str], typer.Option(help="Path to a LoRA checkpoint.")] = None,
    config_path: Annotated[Optional[str], typer.Option(help="Path to a config file.")] = None,
    device: Annotated[str, typer.Option(help="Device to run on.")] = "cuda",
    fuse_lora: Annotated[
        bool, typer.Option("--fuse-lora/--no-fuse-lora", help="Fuse LoRA layers.")
    ] = True,
    bf16: Annotated[bool, typer.Option(help="Use bfloat16.")] = False,
    ssl: Annotated[
        Optional[str], typer.Option(help="Directory containing cert.pem and key.pem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
):
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
        moshi_weight,
        mimi_weight,
        tokenizer,
        lora_weights=lora_weight,
        config_path=config_path,
    )

    log("info", "Loading the codec...")
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "Loading the model...")
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
def generate():
    raise NotImplementedError("WIP")


def main():
    cli_app()


if __name__ == "__main__":
    main()
