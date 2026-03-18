"use client";

import { useHistory } from "@/features/prediction/hooks/useHistory";
import { Button } from "@/components/ui/button";
import { ChevronRight, Trash2, Calendar, Clock, History } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "@/components/ui/toast";
import { useCallback } from "react";

export default function HistoryPage() {
  const { history, clearHistory, isLoaded } = useHistory();

  const handleClear = useCallback(() => {
    if (window.confirm("Are you sure you want to clear all analysis history? This action cannot be undone.")) {
      clearHistory();
      toast("History cleared successfully", "success");
    }
  }, [clearHistory]);

  if (!isLoaded) {
    return (
      <div className="container py-4 md:px-6 max-w-5xl mx-auto space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-4 w-64" />
          </div>
        </div>
        <div className="grid gap-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-16 w-full rounded-md" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="container py-4 px-4 md:px-6 max-w-5xl mx-auto animate-in fade-in duration-300">
      <div className="flex items-center justify-between mb-6">
        <div>
            <h1 className="text-2xl font-bold tracking-tight">Analysis History</h1>
            <p className="text-sm text-muted-foreground">View your past crop recommendations.</p>
        </div>
        {history.length > 0 && (
            <Button 
              variant="destructive" 
              size="sm"
              onClick={handleClear}
              className="font-bold"
            >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear
            </Button>
        )}
      </div>

      <div className="grid gap-2">
        {history.length === 0 ? (
            <div className="text-center py-12 border-2 border-dashed rounded-md bg-muted/10 border-border">
                <History className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
                <h3 className="text-lg font-bold text-muted-foreground mb-1">No history found</h3>
                <p className="text-sm text-muted-foreground/60 mb-4">Your past soil analyses will appear here.</p>
                <Link href="/analyze">
                    <Button size="sm" className="font-bold">Start New Analysis</Button>
                </Link>
            </div>
        ) : (
            history.map((item) => (
                <Link key={item.id} href={`/analyze/result?id=${item.id}`}>
                    <div className="group flex items-center justify-between p-3 rounded-md border border-border bg-card hover:bg-muted/50 transition-colors cursor-pointer">
                        <div className="flex items-center gap-4">
                            <div className="h-10 w-10 rounded-md bg-primary/10 flex items-center justify-center font-bold text-primary text-sm">
                                {Math.round((item.result.predictions?.[0]?.confidence || 0.95) * 100)}%
                            </div>
                            <div className="space-y-0.5">
                                <h3 className="text-sm font-bold group-hover:text-primary transition-colors">
                                    {item.result.consensus_crop}
                                </h3>
                                <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                                    <span className="flex items-center gap-1 font-medium">
                                        <Calendar className="w-3 h-3" />
                                        {new Date(item.timestamp).toLocaleDateString()}
                                    </span>
                                    <span className="flex items-center gap-1 font-medium">
                                        <Clock className="w-3 h-3" />
                                        {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition-transform" />
                    </div>
                </Link>
            ))
        )}
      </div>
    </div>
  );
}
