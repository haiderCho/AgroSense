"use client";

import { motion } from "framer-motion";
import { Sprout, Leaf, Cpu, Activity, ArrowRight, Wheat, Database } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-3.5rem)] px-6 py-8 md:p-12 relative overflow-hidden bg-background selection:bg-emerald-100 selection:text-emerald-900">
      {/* Immersive Background Layer */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
        {/* Dynamic Gradient Mesh */}
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-emerald-500/15 rounded-full blur-[160px] -translate-y-1/2 translate-x-1/3 animate-pulse" />
        <div className="absolute bottom-0 left-0 w-[700px] h-[700px] bg-primary/10 rounded-full blur-[140px] translate-y-1/3 -translate-x-1/4 animate-pulse duration-700" />
        
        {/* Animated Background Icons */}
        <motion.div 
            initial={{ opacity: 0, rotate: -30 }}
            animate={{ opacity: 0.15, rotate: -15, y: [0, 40, 0] }}
            transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -left-20 top-40"
        >
          <Wheat className="w-[450px] h-[450px] text-emerald-600" />
        </motion.div>
        
        <motion.div 
            initial={{ opacity: 0, rotate: 20 }}
            animate={{ opacity: 0.12, rotate: 35, y: [0, -60, 0] }}
            transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -right-20 bottom-40"
        >
          <Sprout className="w-[550px] h-[550px] text-primary" />
        </motion.div>

        {/* Neural Grid Overlay */}
        <div className="absolute inset-0 bg-[radial-gradient(#10b981_0.8px,transparent_0.8px)] [background-size:32px_32px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_20%,transparent_100%)] opacity-[0.03]" />
      </div>

      <div className="text-center max-w-4xl space-y-6 relative z-10 pt-6">
        <motion.div 
            className="space-y-4"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <div className="flex items-center justify-center gap-3 mb-1">
            <motion.div 
                whileHover={{ rotate: 180 }}
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center shadow-inner"
            >
              <Leaf className="w-6 h-6 text-emerald-600" />
            </motion.div>
            <h1 className="text-5xl md:text-8xl font-black tracking-tighter text-foreground drop-shadow-sm">
                Agro<span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-primary">Sense</span>
            </h1>
          </div>
          
          <h2 className="text-xl md:text-3xl font-extrabold tracking-tight text-foreground max-w-2xl mx-auto leading-[1.1]">
            Next-gen <span className="text-emerald-600">crop recommendation</span> engine powered by ensemble intelligence.
          </h2>
          
          <p className="text-sm md:text-lg text-muted-foreground/80 max-w-xl mx-auto leading-relaxed font-medium">
            Harnessing explainable AI to optimize agricultural yield with precision soil diagnostics.
          </p>
        </motion.div>

        <motion.div 
            className="grid grid-cols-1 md:grid-cols-3 gap-5 my-10 text-left"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
        >
            {[
                { icon: Sprout, title: "Precision Farming", desc: "Data-driven sensor optimization for specific regional conditions." },
                { icon: Cpu, title: "AI Ensemble", desc: "Multi-model consensus logic delivers high-accuracy suitability scores." },
                { icon: Activity, title: "Explainable AI", desc: "Transparent SHAP metrics behind every prediction for informed decisions." },
            ].map((feature, i) => (
                <div 
                    key={i}
                    className="group relative p-6 rounded-lg bg-background/40 backdrop-blur-md border border-border/50 hover:border-emerald-500/40 transition-all shadow-sm hover:shadow-md"
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent rounded-lg opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="w-10 h-10 rounded-lg bg-muted/50 flex items-center justify-center mb-4 transition-all group-hover:bg-emerald-500 group-hover:text-white shadow-sm">
                      <feature.icon className="w-5 h-5" />
                    </div>
                    <h3 className="text-base font-bold mb-2 text-foreground tracking-tight">{feature.title}</h3>
                    <p className="text-muted-foreground/80 text-xs leading-relaxed font-medium">{feature.desc}</p>
                </div>
            ))}
        </motion.div>

        <motion.div 
            className="flex flex-col items-center gap-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
        >
          <Link href="/analyze">
            <Button
              size="lg"
              className="px-8 py-5 h-14 text-lg font-black rounded-lg shadow-md hover:shadow-lg active:scale-[0.98] transition-all bg-emerald-600 hover:bg-emerald-700 flex items-center gap-2 group"
            >
                <span>Begin Analysis</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>

          {/* Technology Partners / Ecosystem */}
          <div className="pt-12 pb-6 w-full max-w-2xl">
            <p className="text-[10px] font-black text-muted-foreground/50 uppercase tracking-[0.3em] mb-6">
              Precision Intelligence Ecosystem
            </p>
            <div className="flex items-center justify-center gap-8 md:gap-16 opacity-30">
                <div className="flex items-center gap-2 grayscale hover:grayscale-0 transition-all cursor-default group">
                    <div className="p-2 rounded-lg bg-emerald-500/10 group-hover:bg-emerald-500/20">
                      <Cpu className="w-4 h-4 text-emerald-600" />
                    </div>
                    <span className="text-xs font-bold tracking-tight">AI Core</span>
                </div>
                <div className="flex items-center gap-2 grayscale hover:grayscale-0 transition-all cursor-default group">
                    <div className="p-2 rounded-lg bg-emerald-500/10 group-hover:bg-emerald-500/20">
                      <Database className="w-4 h-4 text-emerald-600" />
                    </div>
                    <span className="text-xs font-bold tracking-tight">Big Data</span>
                </div>
                <div className="flex items-center gap-2 grayscale hover:grayscale-0 transition-all cursor-default group">
                    <div className="p-2 rounded-lg bg-emerald-500/10 group-hover:bg-emerald-500/20">
                      <Leaf className="w-4 h-4 text-emerald-600" />
                    </div>
                    <span className="text-xs font-bold tracking-tight">Agri-Stack</span>
                </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Grounding Footer */}
      <footer className="w-full max-w-6xl mt-12 py-8 px-6 border-t border-border/50 flex flex-col md:flex-row justify-between items-center gap-4 text-muted-foreground relative z-10">
        <div className="flex items-center gap-2 group">
            <div className="w-6 h-6 rounded-md bg-emerald-500/10 flex items-center justify-center">
              <Sprout className="w-3.5 h-3.5 text-emerald-600" />
            </div>
            <span className="text-sm font-black text-foreground tracking-tighter">AgroSense</span>
            <span className="text-[10px] font-bold text-muted-foreground/40 hidden sm:inline">— BUILT FOR PRECISION</span>
        </div>
        
        <div className="flex items-center gap-6 text-[11px] font-bold uppercase tracking-widest opacity-60">
            <Link href="/dashboard" className="hover:text-emerald-600 transition-colors">Dashboard</Link>
            <Link href="/analyze" className="hover:text-emerald-600 transition-colors">Analyze</Link>
            <Link href="/history" className="hover:text-emerald-600 transition-colors">History</Link>
        </div>
        
        <div className="text-[10px] font-bold text-muted-foreground/40">
            © 2026 NEXT-GEN AGRI-ENGINE. ALL RIGHTS RESERVED.
        </div>
      </footer>

      {/* Soft Bottom Anchor Gradient */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-full h-[300px] bg-gradient-to-t from-emerald-500/10 to-transparent blur-[100px] pointer-events-none opacity-40 select-none" aria-hidden="true" />
    </div>
  );
}

