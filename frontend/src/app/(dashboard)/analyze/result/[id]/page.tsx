"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { PredictionResult, useHistory, type PredictionResponse, type HistoryItem } from "@/features/prediction";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function ResultPage() {
  const { id } = useParams();
  const router = useRouter();
  const { history } = useHistory();
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Wait for history to load from localStorage
    if (history.length === 0) {
        const saved = localStorage.getItem("agrosense_history");
        if (saved) {
             const items = JSON.parse(saved) as HistoryItem[];
             const item = items.find((i) => i.id === id);
             if (item) {
                 setResult(item.result);
                 setLoading(false);
                 return;
             }
        }
    } else {
        const item = history.find((i) => i.id === id);
        if (item) {
            setResult(item.result);
            setLoading(false);
        }
    }
    
    // Fallback: If after 500ms we still have nothing, maybe redirect?
    const timer = setTimeout(() => setLoading(false), 500);
    return () => clearTimeout(timer);

  }, [id, history, router]);

  if (loading) {
      return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  }

  if (!result) {
    return (
      <div className="container py-16 text-center">
        <h1 className="text-2xl font-bold mb-4">Result Not Found</h1>
        <p className="text-muted-foreground mb-8">This analysis might have expired or been deleted.</p>
        <Link href="/analyze">
          <Button>Create New Analysis</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="container py-16 px-4 md:px-6 max-w-4xl mx-auto">
      <div className="mb-6 print:hidden">
        <Link href="/analyze" className="inline-flex items-center text-sm text-muted-foreground hover:text-primary transition-colors">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Analyze
        </Link>
      </div>
      
      <PredictionResult 
        result={result} 
        onReset={() => router.push("/analyze")} 
      />
    </div>
  );
}
