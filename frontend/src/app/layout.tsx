import type { Metadata } from "next";
import { Outfit, Space_Mono } from "next/font/google";
import "./globals.css";
import { ThemeToggle } from "@/components/ui/theme-toggle";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  display: 'swap',
});

const spaceMono = Space_Mono({
  weight: ['400', '700'],
  subsets: ["latin"],
  variable: "--font-space-mono",
  display: 'swap',
});

export const metadata: Metadata = {
  title: "AgroSense | AI Crop Recommendation",
  description: "Premium AI-powered crop recommendation system.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${outfit.variable} ${spaceMono.variable} font-sans antialiased bg-background text-foreground relative overflow-x-hidden transition-colors duration-300`}
      >
        <ThemeToggle />
        <div className="texture-overlay dark:opacity-[0.03] opacity-[0.02]" />
        <main className="relative z-10 min-h-screen">
            {children}
        </main>
      </body>
    </html>
  );
}
