"use client";

import { motion } from "framer-motion";
import { Leaf, Activity, History, ArrowRight } from "lucide-react";
import Link from "next/link";
import { useHistory } from "@/features/prediction/hooks/useHistory";

export default function DashboardPage() {
  const { history } = useHistory();

  const stats = [
    {
      label: "Total Analyses",
      value: history.length,
      icon: Activity,
      color: "text-blue-500",
      bg: "bg-blue-500/10",
    },
    {
      label: "Latest Prediction",
      value: history.length > 0 ? history[0].result.predictions[0].crop : "N/A",
      icon: Leaf,
      color: "text-green-500",
      bg: "bg-green-500/10",
    },
    {
      label: "History Saved",
      value: history.length > 0 ? "Yes" : "No",
      icon: History,
      color: "text-purple-500",
      bg: "bg-purple-500/10",
    },
  ];

  return (
    <div className="container py-8 px-4 md:px-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Dashboard</h1>
        <p className="text-muted-foreground">Welcome back to AgroSense. Here&apos;s your system overview.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3 mb-12">
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="p-6 rounded-xl border border-border bg-card/50 backdrop-blur-sm flex items-center gap-4 hover:border-primary/50 transition-colors"
          >
            <div className={`w-12 h-12 rounded-lg ${stat.bg} flex items-center justify-center`}>
              <stat.icon className={`w-6 h-6 ${stat.color}`} />
            </div>
            <div>
              <p className="text-sm text-muted-foreground font-medium">{stat.label}</p>
              <h3 className="text-2xl font-bold">{stat.value}</h3>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="p-8 rounded-2xl bg-gradient-to-br from-primary/10 via-card to-card border border-primary/20 relative overflow-hidden group"
        >
             <div className="relative z-10">
                <h2 className="text-2xl font-bold mb-4">Start New Analysis</h2>
                <p className="text-muted-foreground mb-6 max-w-md">
                    Use our advanced multi-model ensemble to determine the best crop for your soil conditions.
                </p>
                <Link href="/analyze" className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg font-bold hover:bg-primary/90 transition-colors">
                    Analyze Soil <ArrowRight className="w-4 h-4" />
                </Link>
             </div>
             <Leaf className="absolute -right-10 -bottom-10 w-64 h-64 text-primary/5 group-hover:text-primary/10 transition-colors rotate-[-15deg]" />
        </motion.div>

        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
             className="p-8 rounded-2xl border border-border bg-card relative overflow-hidden"
        >
            <h2 className="text-2xl font-bold mb-4">Recent History</h2>
            {history.length > 0 ? (
                <div className="space-y-4">
                    {history.slice(0, 3).map((item) => (
                        <Link key={item.id} href={`/analyze/result/${item.id}`} className="block">
                            <div className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors border border-transparent hover:border-border">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                                        <Leaf className="w-4 h-4 text-primary" />
                                    </div>
                                    <span className="font-medium">{item.result.predictions[0].crop}</span>
                                </div>
                                <span className="text-sm text-muted-foreground">{new Date(item.timestamp).toLocaleDateString()}</span>
                            </div>
                        </Link>
                    ))}
                    <Link href="/history" className="block text-center text-sm text-primary hover:underline mt-4">
                        View All History
                    </Link>
                </div>
            ) : (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground pb-8">
                    <History className="w-12 h-12 mb-4 opacity-20" />
                    <p>No analysis history yet.</p>
                </div>
            )}
        </motion.div>
      </div>
    </div>
  );
}
