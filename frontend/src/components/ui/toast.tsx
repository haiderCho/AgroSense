"use client";

import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

export type ToastType = "success" | "error" | "info";

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

let toastFn: (message: string, type: ToastType) => void;

export const toast = (message: string, type: ToastType = "info") => {
  if (toastFn) toastFn(message, type);
};

export function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    toastFn = (message: string, type: ToastType) => {
      const id = Math.random().toString(36).substring(2, 9);
      setToasts((prev) => [...prev, { id, message, type }]);
      setTimeout(() => removeToast(id), 5000);
    };
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={cn(
              "pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg min-w-[300px] max-w-md",
              t.type === "success" && "bg-green-500/10 border-green-500/20 text-green-500",
              t.type === "error" && "bg-destructive/10 border-destructive/20 text-destructive",
              t.type === "info" && "bg-primary/10 border-primary/20 text-primary"
            )}
          >
            {t.type === "success" && <CheckCircle2 className="w-5 h-5 shrink-0" />}
            {t.type === "error" && <AlertCircle className="w-5 h-5 shrink-0" />}
            {t.type === "info" && <Info className="w-5 h-5 shrink-0" />}
            
            <p className="text-sm font-medium flex-1">{t.message}</p>
            
            <button
              onClick={() => removeToast(t.id)}
              className="p-1 hover:bg-foreground/5 rounded-md transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
