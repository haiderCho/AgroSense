"use client";

import React, { useMemo } from "react";
import { Brain, Lightbulb } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface InsightGeneratorProps {
  crop: string;
  explanation: Record<string, number>;
}

export const InsightGenerator = React.memo(({ crop, explanation }: InsightGeneratorProps) => {
  const topFactors = useMemo(() => {
    return Object.entries(explanation)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3);
  }, [explanation]);

  return (
    <Card className="bg-primary/5 border-primary/10 overflow-hidden relative shadow-sm">
      <CardHeader>
        <CardTitle className="text-lg font-bold flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" />
          AI Analysis Insights
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs leading-relaxed">
          The models recommend <span className="font-bold text-primary underline decoration-1 underline-offset-2">{crop}</span> based on these factors:
        </p>

        <div className="space-y-2">
          {topFactors.map(([factor, score], idx) => (
            <div 
              key={factor}
              className="flex items-center gap-3 bg-card p-2 rounded-md border border-border"
            >
              <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center font-bold text-primary text-[10px] shrink-0">
                {idx + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex justify-between items-center mb-0.5">
                  <span className="text-[10px] font-bold uppercase tracking-tight truncate mr-2">{factor.replace("_", " ")}</span>
                  <span className="text-[9px] font-bold text-primary shrink-0">{Math.round(score * 100)}%</span>
                </div>
                <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-primary transition-all duration-500"
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-start gap-2 pt-1 text-[9px] text-muted-foreground p-2 rounded bg-muted/20 border border-dashed border-border">
          <Lightbulb className="w-3 h-3 text-amber-500 shrink-0" />
          <p className="leading-tight">
            Recommendation derived from an ensemble of crossover ML architectures.
          </p>
        </div>
      </CardContent>
    </Card>
  );
});

InsightGenerator.displayName = "InsightGenerator";
