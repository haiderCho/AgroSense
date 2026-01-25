"use client";

import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip, Cell } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PredictionResponse } from "../types";

interface ConfidenceChartProps {
  predictions: PredictionResponse["predictions"];
}

export function ConfidenceChart({ predictions }: ConfidenceChartProps) {
  // Transform data for chart
  const data = predictions.map((p) => ({
    name: p.model.toUpperCase(),
    confidence: Math.round(p.confidence * 100),
    crop: p.crop,
  }));

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-lg">Model Consensus</CardTitle>
        <CardDescription>Confidence levels across ensemble models</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[200px] w-full">
          <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
            <BarChart data={data} layout="vertical" margin={{ left: 10 }}>
              <defs>
                <linearGradient id="confidenceGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.7} />
                  <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={1} />
                </linearGradient>
              </defs>
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis 
                dataKey="name" 
                type="category" 
                width={120} 
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))", fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip 
                cursor={{ fill: "hsl(var(--muted)/0.2)" }}
                contentStyle={{ 
                    backgroundColor: "hsl(var(--popover)/0.9)", 
                    borderColor: "hsl(var(--border))",
                    borderRadius: "8px",
                    backdropFilter: "blur(8px)"
                }}
                itemStyle={{ color: "hsl(var(--foreground))" }}
              />
              <Bar dataKey="confidence" radius={[0, 4, 4, 0]}>
                {data.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.confidence > 80 ? "url(#confidenceGradient)" : "hsl(var(--muted-foreground)/0.4)"} 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
