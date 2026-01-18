"use client";

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface FeatureImportanceChartProps {
  explanation?: Record<string, number>;
}

export function FeatureImportanceChart({
  explanation,
}: FeatureImportanceChartProps) {
  if (!explanation || Object.keys(explanation).length === 0) return null;

  // Transform and sort by importance
  const data = Object.entries(explanation)
    .map(([feature, value]) => ({
      feature: feature.charAt(0).toUpperCase() + feature.slice(1),
      value: Math.round(Math.abs(value) * 100), // Convert to percentage
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-lg">Decision Factors (xAI)</CardTitle>
        <CardDescription>
          Key parameters influencing the recommendation
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[220px] w-full">
          <ResponsiveContainer
            width="100%"
            height="100%"
            minWidth={0}
            minHeight={0}
          >
            <BarChart
              data={data}
              layout="vertical"
              margin={{ left: 10, right: 20 }}
            >
              <XAxis
                type="number"
                domain={[0, 100]}
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                tickFormatter={(v) => `${v}%`}
              />
              <YAxis
                dataKey="feature"
                type="category"
                width={110}
                tick={{
                  fill: "hsl(var(--foreground))",
                  fontSize: 11,
                  fontWeight: 500,
                }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "hsl(var(--muted)/0.3)" }}
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  borderColor: "hsl(var(--border))",
                  borderRadius: "8px",
                }}
                formatter={(value: number | undefined) => [
                  `${value ?? 0}%`,
                  "Impact",
                ]}
              />
              <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={20}>
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      index === 0
                        ? "hsl(var(--primary))"
                        : "hsl(var(--accent)/0.6)"
                    }
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
