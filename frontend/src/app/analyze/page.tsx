"use client";

import { useState } from "react";
import { PredictionForm, PredictionResult } from "@/features/prediction/components";
import { PredictionResponse } from "@/features/prediction/types";
import { AnimatePresence, motion } from "framer-motion";

export default function AnalyzePage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);

  return (
    <div className="container py-16 px-4 md:px-6 relative z-10 max-w-4xl mx-auto">
      <div className="flex flex-col items-center space-y-4 text-center mb-8">
        <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
          {result ? "Analysis Report" : "New Analysis"}
        </h1>
        <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
          {result 
            ? "AI-generated recommendations based on your soil profile." 
            : "Configure soil parameters to generate a new crop suitability report."}
        </p>
      </div>

      <AnimatePresence mode="wait">
        {result ? (
            <motion.div
                key="result"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
            >
                <PredictionResult 
                    result={result} 
                    onReset={() => setResult(null)} 
                />
            </motion.div>
        ) : (
            <motion.div
                key="form"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
            >
                <PredictionForm onAnalysisComplete={setResult} />
            </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
