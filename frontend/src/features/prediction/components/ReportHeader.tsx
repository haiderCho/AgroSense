"use client";

import { Leaf } from "lucide-react";

export function ReportHeader() {
  return (
    <div className="hidden print:flex flex-col mb-8 border-b border-border pb-6">
      <div className="flex justify-between items-start">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary rounded-lg text-primary-foreground print-color-exact">
            <Leaf className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground tracking-tight">AgroSense</h1>
            <p className="text-sm text-muted-foreground font-medium uppercase tracking-wider">AI Crop Analysis Report</p>
          </div>
        </div>
        <div className="text-right space-y-1">
          <div className="inline-block px-3 py-1 bg-muted rounded text-xs font-mono mb-2 print-color-exact">
            CONFIDENTIAL
          </div>
          <p className="text-sm text-muted-foreground">Generated on:</p>
          <p className="text-lg font-mono font-medium text-foreground">
            {new Date().toLocaleDateString(undefined, { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            })}
          </p>
        </div>
      </div>
    </div>
  );
}
