"use client";

import { motion } from "framer-motion";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle,
  CardDescription 
} from "@/components/ui/card";
import { 
  Sprout, 
  Activity, 
  History, 
  MapPin, 
  Droplets, 
  Thermometer,
  ChevronRight,
  TrendingUp,
  AlertCircle
} from "lucide-react";
import { useHistory } from "@/features/prediction";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function DashboardPage() {
  const { history, isLoaded } = useHistory();
  
  // Data Aggregation Logic
  const hasHistory = history.length > 0;
  const recentHistory = history.slice(0, 5);

  const stats = (() => {
    if (!hasHistory) return [
        { label: "Total Analyses", value: "0", icon: Activity, color: "text-blue-500" },
        { label: "Top Crop", value: "N/A", icon: Sprout, color: "text-green-500" },
        { label: "Avg Confidence", value: "0%", icon: TrendingUp, color: "text-orange-500" },
        { label: "Mean Soil pH", value: "0.0", icon: Droplets, color: "text-cyan-500" },
    ];

    const totalAnalyses = history.length;
    
    // Most common crop
    const cropCounts = history.reduce((acc, item) => {
        const topCrop = item.result.predictions[0]?.crop || "Unknown";
        acc[topCrop] = (acc[topCrop] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);
    const mostCommonCrop = Object.entries(cropCounts).sort((a, b) => b[1] - a[1])[0][0];

    // Avg Confidence
    const avgConfidence = Math.round(
        (history.reduce((acc, item) => acc + (item.result.predictions[0]?.confidence || 0), 0) / totalAnalyses) * 100
    );

    // Avg pH
    const avgPH = (history.reduce((acc, item) => acc + (item.input.ph || 0), 0) / totalAnalyses).toFixed(1);

    return [
      { label: "Total Analyses", value: totalAnalyses.toString(), icon: Activity, color: "text-blue-500" },
      { label: "Top Recommendation", value: mostCommonCrop.charAt(0).toUpperCase() + mostCommonCrop.slice(1), icon: Sprout, color: "text-green-500" },
      { label: "Model Confidence", value: `${avgConfidence}%`, icon: TrendingUp, color: "text-orange-500" },
      { label: "Avg Soil pH", value: avgPH, icon: Thermometer, color: "text-cyan-500" },
    ];
  })();

  const insights = (() => {
    if (!hasHistory) return {
        trend: "Start an analysis to see your personalized soil trends and crop recommendations.",
        alert: "No critical soil alerts at this time. Perform a diagnostic to check local conditions.",
        consistency: "N/A",
        risk: "Low"
    };

    const latest = history[0];
    const topCrop = latest.result.predictions[0]?.crop;
    const confidence = latest.result.predictions[0]?.confidence;

    const trend = confidence > 0.8 
        ? `High consensus for ${topCrop}. Your soil metrics strongly favor this cultivation cycle.`
        : `Moderate suitablity for ${topCrop}. Consider adjusting Nitrogen levels for better results.`;

    const avgN = history.reduce((acc, item) => acc + (item.input.N || 0), 0) / history.length;
    let alert = "Soil metrics are within optimal range for your recent crop matches.";
    let risk = "Low";

    if (latest.input.N < avgN * 0.8) {
        alert = "Current Nitrogen levels are 20% lower than your historical average. Fertilization recommended.";
        risk = "Moderate";
    } else if (latest.input.ph < 5.5 || latest.input.ph > 7.5) {
        alert = "Recent pH levels show significant deviation. Soil neutralizing might be required.";
        risk = "High";
    }

    return {
        trend,
        alert,
        consistency: confidence > 0.8 ? "High" : "Moderate",
        risk
    };
  })();

  if (!isLoaded) return <div className="p-8 animate-pulse text-muted-foreground">Loading Analytics...</div>;

  return (
    <div className="container py-8 px-6 max-w-7xl mx-auto">
      <div className="space-y-8 animate-in fade-in duration-500">
        {/* Welcome Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b pb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">Agricultural Intelligence</h1>
            <p className="text-muted-foreground text-base max-w-2xl font-medium">
              Data-driven insights derived from your {history.length} most recent soil diagnostic reports.
            </p>
          </div>
          <div className="flex gap-3">
             <Link href="/analyze">
                <Button className="font-extrabold px-8 rounded-lg shadow-lg shadow-primary/10 transition-all hover:scale-105 active:scale-95">
                    Start New Analysis
                </Button>
             </Link>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, i) => (
            <Card key={i} className="border-border shadow-sm hover:shadow-md transition-all group rounded-lg">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className={cn("p-3 rounded-lg bg-muted/50 transition-colors group-hover:bg-primary/5", stat.color)}>
                    <stat.icon className="w-5 h-5 transition-transform group-hover:scale-110" />
                  </div>
                  <div>
                    <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">{stat.label}</p>
                    <h3 className="text-xl font-bold tracking-tight">{stat.value}</h3>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Activity */}
          <Card className="lg:col-span-2 border-border shadow-sm rounded-lg overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between border-b bg-muted/5 py-4 px-6">
              <div className="space-y-0.5">
                <CardTitle className="text-lg font-bold">Recent Analyses</CardTitle>
                <CardDescription className="text-xs">Your latest diagnostic history</CardDescription>
              </div>
              <Link href="/history">
                <Button variant="ghost" size="sm" className="text-xs font-bold hover:text-primary">View Full History</Button>
              </Link>
            </CardHeader>
            <CardContent className="p-0">
               {hasHistory ? (
                 <div className="divide-y divide-border/40">
                    {recentHistory.map((item, i) => (
                      <Link 
                        key={i} 
                        href={`/analyze/result?id=${item.id}`}
                        className="flex items-center justify-between p-5 hover:bg-muted/30 transition-colors group"
                      >
                        <div className="flex items-center gap-4">
                          <div className={cn(
                            "w-10 h-10 rounded-full flex items-center justify-center transition-all",
                            i === 0 ? "bg-primary/10 text-primary shadow-sm" : "bg-muted text-muted-foreground opacity-70"
                          )}>
                            <Sprout className="w-5 h-5" />
                          </div>
                          <div>
                            <p className="text-base font-bold text-foreground capitalize">{item.result.predictions[0]?.crop}</p>
                            <p className="text-xs text-muted-foreground font-medium">{new Date(item.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="text-right hidden sm:block">
                                <p className="text-xs font-bold text-muted-foreground uppercase tracking-tighter">Confidence</p>
                                <p className="text-sm font-black tabular-nums">{Math.round((item.result.predictions[0]?.confidence || 0) * 100)}%</p>
                            </div>
                            <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-transform group-hover:translate-x-1" />
                        </div>
                      </Link>
                    ))}
                 </div>
               ) : (
                 <div className="p-16 text-center">
                    <History className="w-16 h-16 text-muted-foreground/20 mx-auto mb-4" />
                    <p className="text-muted-foreground font-medium">No analysis history found.</p>
                    <p className="text-xs text-muted-foreground/60 mt-1">Your diagnostics will appear here after your first run.</p>
                 </div>
               )}
            </CardContent>
          </Card>

          {/* Intelligent Insights */}
          <div className="space-y-6">
            <Card className="border-border shadow-sm bg-primary/5 rounded-lg">
                <CardHeader className="pb-2">
                    <div className="flex items-center gap-2 text-primary">
                        <TrendingUp className="w-5 h-5" />
                        <CardTitle className="text-base font-bold">Dynamic Insights</CardTitle>
                    </div>
                </CardHeader>
                <CardContent className="space-y-4">
                    <p className="text-sm font-medium leading-relaxed italic opacity-80">
                        "{insights.trend}"
                    </p>
                    <div className="grid grid-cols-2 gap-4 pt-2">
                      <div className="p-3 rounded-lg border border-border bg-background shadow-sm">
                        <p className="text-[9px] font-black text-muted-foreground uppercase mb-1 tracking-wider">Stability</p>
                        <p className="text-lg font-black text-primary">{insights.consistency}</p>
                      </div>
                      <div className="p-3 rounded-lg border border-border bg-background shadow-sm">
                        <p className="text-[9px] font-black text-muted-foreground uppercase mb-1 tracking-wider">Trend Strength</p>
                        <p className={cn(
                            "text-lg font-black",
                            insights.risk === "High" ? "text-destructive" : (insights.risk === "Moderate" ? "text-orange-500" : "text-green-600")
                        )}>{insights.risk}</p>
                      </div>
                    </div>
                </CardContent>
            </Card>

            <Card className="border-border shadow-sm rounded-lg overflow-hidden">
                <CardHeader className="pb-2 bg-orange-500/5">
                    <div className="flex items-center gap-2 text-orange-600">
                        <AlertCircle className="w-5 h-5" />
                        <CardTitle className="text-base font-bold">Smart Alert</CardTitle>
                    </div>
                </CardHeader>
                <CardContent className="pt-4 pb-6">
                    <p className="text-sm text-muted-foreground font-medium leading-relaxed">
                        {insights.alert}
                    </p>
                    {hasHistory && (
                        <Link href="/analyze" className="mt-6 block">
                            <Button variant="outline" className="w-full h-10 rounded-lg text-xs font-bold hover:bg-orange-500 hover:text-white transition-all">Refine Parameters</Button>
                        </Link>
                    )}
                </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

function cn(...inputs: any[]) {
    return inputs.filter(Boolean).join(" ");
}
