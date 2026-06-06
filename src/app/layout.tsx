import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "SepsisScope — Retinal AVR Analysis",
  description:
    "Non-invasive fundus image analysis for arteriovenous ratio (AVR) computation — a microvascular biomarker associated with sepsis, hypertension, and cardiovascular risk.",
  keywords: ["fundus", "AVR", "arteriovenous ratio", "sepsis", "retinal analysis", "microvascular"],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-slate-50 font-sans">{children}</body>
    </html>
  );
}
