import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hibiki-Zero",
  description: "Kyutai's real-time speech translation model",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
