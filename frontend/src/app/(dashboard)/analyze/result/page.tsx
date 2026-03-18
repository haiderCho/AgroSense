"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { PredictionResult, useHistory, type PredictionResponse, type HistoryItem } from "@/features/prediction";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Loader2 } from "lucide-react";
import Link from "next/link";
import { Suspense } from "react";

function ResultPageContent() {
  const params = useSearchParams();
  const router = useRouter();
  const id = params.get("id");
  const { history } = useHistory();

  // Start with null - defer all localStorage reads to useEffect to avoid hydration mismatch
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!id) {
        // No ID? Invalid.
        setIsLoading(false); 
        return;
    }

    if (result) {
        setIsLoading(false); 
        return;
    }

    // Check from in-memory history first (when useHistory has hydrated)
    if (history.length > 0) {
        const found = history.find(h => h.id === id);
        if (found) {
            setResult(found.result);
            setIsLoading(false);
            return;
        }
    }

    // Fallback to disk
    const saved = localStorage.getItem("agrosense_history");
    if (saved) {
        try {
            const items = JSON.parse(saved) as HistoryItem[];
            const found = items.find(i => i.id === id);
            if (found) {
                setResult(found.result);
                setIsLoading(false);
            } else {
                setIsLoading(false); // Valid ID format but not in DB
            }
        } catch {
             setIsLoading(false);
        }
    } else {
        setIsLoading(false);
    }
  }, [id, result, history]);

  if (isLoading) {
      return (
         <div className="min-h-[50vh] flex flex-col items-center justify-center text-muted-foreground">
            <Loader2 className="w-8 h-8 animate-spin mb-4 text-primary" />
            <p>Retrieving Analysis...</p>
         </div>
      );
  }

  if (!result) {
    return (
      <div className="container py-12 text-center">
        <h1 className="text-xl font-bold mb-2">Result Not Found</h1>
        <p className="text-sm text-muted-foreground mb-6">
            We couldn&apos;t locate this analysis record. It may have been cleared from your local history.
        </p>
        <Link href="/analyze">
          <Button variant="default" size="sm">
             Create New Analysis
          </Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="container py-6 px-4 md:px-6 max-w-3xl mx-auto">
      <div className="mb-4 print:hidden">
        <Link href="/analyze" className="inline-flex items-center text-xs text-muted-foreground hover:text-primary transition-colors">
          <ArrowLeft className="w-3 h-3 mr-1" />
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

export default function ResultPage() {
    return (
        <Suspense fallback={<div className="min-h-screen grid place-items-center"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>}>
            <ResultPageContent />
        </Suspense>
    );
}
