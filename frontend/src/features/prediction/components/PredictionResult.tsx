"use client";
import { motion } from "framer-motion";
import { PredictionResponse } from "../types";
import { Card, CardContent } from "@/components/ui/card";
import { CheckCircle2, AlertTriangle, Award, Medal, Trophy, Printer } from "lucide-react";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import { useRef } from "react";
import { useReactToPrint } from "react-to-print";
import { ReportHeader } from "./ReportHeader";

// Lazy load heavy chart components
const ConfidenceChart = dynamic(() => import("./ConfidenceChart").then(mod => mod.ConfidenceChart), {
  loading: () => <div className="h-[200px] w-full animate-pulse bg-muted rounded-lg" />
});

const FeatureImportanceChart = dynamic(() => import("./FeatureImportanceChart").then(mod => mod.FeatureImportanceChart), {
    loading: () => <div className="h-[220px] w-full animate-pulse bg-muted rounded-lg" />
});

const SimulationLab = dynamic(() => import("./SimulationLab").then(mod => mod.SimulationLab), {
    loading: () => <div className="h-[400px] w-full animate-pulse bg-muted rounded-lg" />
});

const InsightGenerator = dynamic(() => import("./InsightGenerator").then(mod => mod.InsightGenerator));

interface PredictionResultProps {
  result: PredictionResponse;
  onReset: () => void;
}

export function PredictionResult({ result, onReset }: PredictionResultProps) {
  const componentRef = useRef<HTMLDivElement>(null);

  const handlePrint = useReactToPrint({
    contentRef: componentRef,
    documentTitle: `AgroSense-Report-${new Date().toISOString().split('T')[0]}`,
    onAfterPrint: () => console.log("Report exported successfully"),
  });

  // Handle empty predictions
  if (!result.predictions || result.predictions.length === 0) {
    return (
      <Card className="border-destructive/50 p-8 text-center">
        <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-destructive" />
        <h2 className="text-xl font-bold mb-2">No Predictions Available</h2>
        <p className="text-muted-foreground mb-4">
          The backend returned no model predictions. Ensure models are trained and loaded.
        </p>
        <button onClick={onReset} className="text-primary underline">
          Try Again
        </button>
      </Card>
    );
  }



  // Sort predictions by confidence
  const sortedPredictions = [...result.predictions].sort((a, b) => b.confidence - a.confidence);
  
  // Get top 3 model predictions (regardless of crop)
  const top3Predictions = sortedPredictions.slice(0, 3);
  
  // Check if there's strong consensus (all models agree)
  const allCrops = result.predictions.map(p => p.crop);
  const isStrongConsensus = new Set(allCrops).size === 1;

  const topPrediction = sortedPredictions[0];

  const rankIcons = [
    <Trophy key="1" className="w-6 h-6 text-yellow-500" />,
    <Medal key="2" className="w-5 h-5 text-slate-400" />,
    <Award key="3" className="w-5 h-5 text-amber-700" />,
  ];

  const rankColors = [
    "border-primary/50 bg-primary/5 shadow-[0_0_30px_-5px_rgba(34,197,94,0.2)]",
    "border-border bg-card/50",
    "border-border bg-card/30",
  ];

  return (
    <div ref={componentRef} className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 print:space-y-6 print:container print:py-8">
      {/* Print Header - Visible only in Print */}
      <ReportHeader />

      {/* Header Actions */}
      <div className="flex justify-between items-center print:hidden">
         <h2 className="text-2xl font-bold">Analysis Results</h2>
         <Button variant="outline" size="sm" onClick={handlePrint} className="gap-2">
            <Printer className="w-4 h-4" /> Export Report
         </Button>
      </div>

      {/* Consensus Banner */}
      {isStrongConsensus && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-center justify-center gap-3 p-4 rounded-lg bg-primary/10 border border-primary/30"
        >
          <CheckCircle2 className="w-6 h-6 text-primary" />
          <span className="text-primary font-medium">
            Strong Consensus: All {result.predictions.length} models agree on <strong className="capitalize">{topPrediction.crop}</strong>
          </span>
        </motion.div>
      )}

      {/* Top 3 Model Predictions */}
      <div>
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4 flex justify-between items-center">
          <span>Top 3 Model Consensus</span>
          <span className="text-xs normal-case bg-secondary px-2 py-1 rounded text-foreground/70">
            Ensemble Agreement: {isStrongConsensus ? 'High' : 'Moderate'}
          </span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {top3Predictions.map((pred, index) => (
            <motion.div
              key={`${pred.model}-${index}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="print:break-inside-avoid"
            >
              <Card className={`h-full transition-all hover:scale-[1.02] ${rankColors[index]} border-opacity-50`}>
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {rankIcons[index]}
                      <span className="text-xs font-mono text-muted-foreground">
                        #{index + 1}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-2xl font-bold font-mono">
                        {Math.round(pred.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mb-2 font-mono uppercase">
                    {pred.model}
                  </p>
                  <h4 className={`text-2xl font-bold tracking-tight capitalize ${index === 0 ? 'text-primary' : 'text-foreground'}`}>
                    {pred.crop}
                  </h4>
                  {index === 0 && (
                    <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> Best Match
                    </p>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* AI Insights Block */}
      <div className="print:break-inside-avoid">
         <InsightGenerator 
            crop={topPrediction.crop} 
            confidence={topPrediction.confidence} 
            explanation={topPrediction.explanation} 
         />
      </div>

      {/* Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="print:break-inside-avoid">
            <ConfidenceChart predictions={result.predictions} />
        </div>
        <div className="print:break-inside-avoid">
            {topPrediction.explanation && Object.keys(topPrediction.explanation).length > 0 ? (
            <FeatureImportanceChart explanation={topPrediction.explanation} />
            ) : (
            <Card className="h-full flex items-center justify-center p-6 text-muted-foreground border-dashed bg-muted/20">
                <div className="text-center">
                <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>xAI Data Unavailable</p>
                </div>
            </Card>
            )}
        </div>
      </div>

      {/* What-If Simulator (Interactive) */}
      <div className="print:hidden pt-4">
         <SimulationLab />
      </div>

      <div className="pt-4 text-center print:hidden">
        <button 
          onClick={onReset}
          className="text-sm text-muted-foreground hover:text-primary transition-colors underline decoration-dotted underline-offset-4"
        >
          Analyze another sample
        </button>
      </div>
    </div>
  );
}
