"use client";

import { useHistory } from "@/features/prediction/hooks/useHistory";

import { Button } from "@/components/ui/button";
import { ChevronRight, Trash2, Calendar, Clock } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";

export default function HistoryPage() {
  const { history, clearHistory } = useHistory();

  return (
    <div className="container py-8 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">Analysis History</h1>
            <p className="text-muted-foreground">View your past crop recommendations.</p>
        </div>
        {history.length > 0 && (
            <Button variant="destructive" onClick={() => {
                if(confirm("Are you sure you want to clear all history?")) clearHistory();
            }}>
                <Trash2 className="w-4 h-4 mr-2" />
                Clear History
            </Button>
        )}
      </div>

      <div className="grid gap-4">
        {history.length === 0 ? (
            <div className="text-center py-20 border rounded-xl bg-muted/20">
                <p className="text-muted-foreground">No history found.</p>
                <Link href="/analyze" className="mt-4 inline-block">
                    <Button>Start New Analysis</Button>
                </Link>
            </div>
        ) : (
            history.map((item, i) => (
                <motion.div
                    key={item.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                >
                    <Link href={`/analyze/result?id=${item.id}`}>
                        <div className="group flex items-center justify-between p-6 rounded-xl border border-border bg-card hover:border-primary/50 hover:shadow-md transition-all">
                            <div className="flex items-center gap-6">
                                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                                    <span className="text-lg font-bold text-primary">
                                        {Math.round(item.result.predictions[0].confidence * 100)}%
                                    </span>
                                </div>
                                <div>
                                    <h3 className="text-lg font-bold group-hover:text-primary transition-colors">
                                        {item.result.predictions[0].crop}
                                    </h3>
                                    <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                                        <span className="flex items-center gap-1">
                                            <Calendar className="w-3 h-3" />
                                            {new Date(item.timestamp).toLocaleDateString()}
                                        </span>
                                        <span className="flex items-center gap-1">
                                            <Clock className="w-3 h-3" />
                                            {new Date(item.timestamp).toLocaleTimeString()}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                        </div>
                    </Link>
                </motion.div>
            ))
        )}
      </div>
    </div>
  );
}
