"use client";

import { useRouter } from "next/navigation";
import { PredictionForm, useHistory, type PredictionResponse } from "@/features/prediction";

export default function AnalyzePage() {
  const router = useRouter();
  const { addToHistory } = useHistory();

  const handleAnalysisComplete = (data: Record<string, number>, result: PredictionResponse) => {
    const id = addToHistory(data, result);
    router.push(`/analyze/result?id=${id}`);
  };

  return (
    <div className="container py-8 px-6 md:px-10 w-full max-w-7xl mx-auto">
      <div className="flex flex-col items-center space-y-4 text-center mb-8">
        <h1 className="text-4xl font-black tracking-tight text-foreground">
           New Analysis
        </h1>
        <p className="mx-auto max-w-[700px] text-muted-foreground text-lg leading-relaxed">
           Configure soil parameters to generate a new crop suitability report powered by AI ensemble models.
        </p>
      </div>

      <div className="max-w-3xl mx-auto">
          <PredictionForm onSuccess={handleAnalysisComplete} />
      </div>
    </div>
  );
}
