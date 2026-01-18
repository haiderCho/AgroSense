"use client";

import { motion } from "framer-motion";
import { ArrowRight, Leaf, Cpu, Activity, Sprout, Wheat, Droplets } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] p-8 relative overflow-hidden">
      {/* Agricultural Background Elements */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.05 }}
          transition={{ duration: 2 }}
          className="absolute -left-20 top-20"
        >
          <Wheat className="w-96 h-96 text-primary rotate-[-30deg]" />
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.05 }}
          transition={{ duration: 2, delay: 0.5 }}
          className="absolute -right-20 bottom-20"
        >
          <Sprout className="w-80 h-80 text-primary rotate-[20deg]" />
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.03 }}
          transition={{ duration: 2, delay: 1 }}
          className="absolute right-1/4 top-10"
        >
          <Droplets className="w-40 h-40 text-blue-500" />
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="text-center max-w-4xl space-y-8 relative z-10"
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium font-mono mb-8">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
          </span>
          SYSTEM ONLINE v2.0
        </div>

        {/* Green AgroSense Title with Leaf Icon */}
        <div className="flex items-center justify-center gap-4">
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: "spring", stiffness: 200, damping: 15, delay: 0.3 }}
          >
            <Leaf className="w-12 h-12 md:w-16 md:h-16 text-primary" />
          </motion.div>
          <h1 className="text-6xl md:text-8xl font-bold tracking-tighter text-primary">
            AgroSense
          </h1>
        </div>
        
        <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
          Next-generation <span className="text-primary font-medium">crop recommendation</span> engine powered by multi-model ensemble learning and explainable AI metrics.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 my-16 text-left">
            {[
                { icon: Sprout, title: "Precision Farming", desc: "Optimized for soil specificity and regional conditions" },
                { icon: Cpu, title: "AI Ensemble", desc: "Multi-model consensus for reliable predictions" },
                { icon: Activity, title: "xAI Insights", desc: "Transparent factors behind every recommendation" },
            ].map((feature, i) => (
                <motion.div 
                    key={i}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 + (i * 0.1) }}
                    className="p-6 rounded-xl bg-card/80 backdrop-blur-sm border border-border hover:border-primary/50 transition-all group hover:shadow-[0_0_30px_-10px_rgba(34,197,94,0.3)]"
                >
                    <feature.icon className="w-8 h-8 text-primary mb-4 group-hover:scale-110 transition-transform" />
                    <h3 className="text-lg font-bold mb-2 font-mono text-foreground">{feature.title}</h3>
                    <p className="text-muted-foreground text-sm">{feature.desc}</p>
                </motion.div>
            ))}
        </div>

        <div className="flex justify-center gap-4">
          <Link href="/analyze">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center gap-2 px-8 py-4 bg-primary text-primary-foreground rounded-lg font-bold text-lg hover:bg-primary/90 transition-colors shadow-[0_0_30px_-5px_rgba(34,197,94,0.4)]"
            >
                <Sprout className="w-5 h-5" />
                Start Analysis <ArrowRight className="w-5 h-5" />
            </motion.button>
          </Link>
        </div>
      </motion.div>
    </div>
  );
}
