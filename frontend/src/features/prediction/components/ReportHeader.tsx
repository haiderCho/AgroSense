"use client";

import { Leaf } from "lucide-react";

export function ReportHeader() {
  return (
    <div className="hidden print:flex flex-col mb-4 border-b border-border pb-4">
      <div className="flex justify-between items-start">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-primary rounded text-primary-foreground print-color-exact">
            <Leaf className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground tracking-tight">AgroSense</h1>
            <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Analysis Report</p>
          </div>
        </div>
        <div className="text-right space-y-0.5">
          <p className="text-[10px] text-muted-foreground uppercase font-bold">Reported on</p>
          <p className="text-sm font-mono font-medium text-foreground">
            {new Date().toLocaleDateString(undefined, { 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric'
            })}
          </p>
        </div>
      </div>
    </div>
  );
}
