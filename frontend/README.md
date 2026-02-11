## Hibiki-Zero frontend

First, [install `pnpm`](https://pnpm.io/installation) if you don't have it:
```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

If you don't have `node` either, get it using:

```bash
pnpm env use --global lts
```

Then install using `pnpm install`.

To run in dev mode, starting a development server that will auto-reload if you change the source files:
```bash
pnpm dev
```

To get a static build that you can serve using a webserver:
```bash
STATIC_EXPORT=1 pnpm next build
```

For either of these, you can specify the host using `--host` and the port using `--port`.