"use client";

import { useRouter } from "next/navigation";
import { PredictionForm, useHistory, type PredictionResponse } from "@/features/prediction";
import { motion } from "framer-motion";

export default function AnalyzePage() {
  const router = useRouter();
  const { addToHistory } = useHistory();

  const handleAnalysisComplete = (data: PredictionResponse) => {
    const id = addToHistory(data);
    router.push(`/analyze/result?id=${id}`);
  };

  return (
    <div className="container py-8 px-4 md:px-8 w-full max-w-7xl mx-auto">
      
      <div className="flex flex-col items-center space-y-4 text-center mb-8">
        <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
           New Analysis
        </h1>
        <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
           Configure soil parameters to generate a new crop suitability report.
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="space-y-12"
      >
          <PredictionForm onAnalysisComplete={handleAnalysisComplete} />
      </motion.div>
    </div>
  );
}
