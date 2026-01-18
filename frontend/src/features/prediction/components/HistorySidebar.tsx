import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { History, X, ChevronRight } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { HistoryItem } from "../hooks/useHistory";

interface HistorySidebarProps {
  history: HistoryItem[];
  onClear: () => void;
}

export function HistorySidebar({ history, onClear }: HistorySidebarProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Toggle Button */}
      <Button
        variant="outline"
        size="icon"
        className="fixed right-4 top-24 z-40 bg-background/80 backdrop-blur-md shadow-lg border-primary/20 hover:border-primary transition-all print:hidden"
        onClick={() => setIsOpen(true)}
      >
        <History className="w-5 h-5 text-primary" />
      </Button>

      {/* Backdrop */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
            className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm print:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed right-0 top-0 bottom-0 z-50 w-full max-w-sm border-l bg-card shadow-2xl print:hidden"
          >
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center gap-2">
                <History className="w-5 h-5 text-primary" />
                <h2 className="text-lg font-bold">Recent Analyses</h2>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)}>
                <X className="w-5 h-5" />
              </Button>
            </div>

            <div className="p-6 overflow-y-auto h-[calc(100vh-80px)] space-y-4">
              {history.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  <p>No history yet.</p>
                  <p className="text-xs mt-2">Run a prediction to see it here.</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {history.map((item) => (
                    <Link
                      key={item.id}
                      href={`/analyze/result?id=${item.id}`}
                      onClick={() => setIsOpen(false)}
                      className="w-full flex items-center justify-between p-4 rounded-lg bg-muted/50 border border-transparent hover:border-primary/30 hover:bg-muted transition-all text-left group"
                    >
                      <div>
                        <div className="font-bold text-foreground group-hover:text-primary transition-colors">
                          {item.label.split(' (')[0]}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1 font-mono">
                          {new Date(item.timestamp).toLocaleDateString()} • {new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors opacity-0 group-hover:opacity-100" />
                    </Link>
                  ))}
                  
                  <div className="pt-4 border-t">
                    <Button 
                        variant="ghost" 
                        size="sm" 
                        className="w-full text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => {
                            if(confirm("Clear all history?")) onClear();
                        }}
                    >
                        Clear History
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
