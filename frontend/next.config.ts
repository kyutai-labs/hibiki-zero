import type { NextConfig } from "next";

const staticExport = ["1", "true"].includes(
  process.env.STATIC_EXPORT?.trim().toLowerCase() || "",
);

const nextConfig: NextConfig = {
  // A static export doesn't work with the hot reloading of `pnpm run dev` so we only use it for production builds
  output: staticExport ? "export" : undefined,
};

export default nextConfig;
