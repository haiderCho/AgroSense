"use client";

import { useState, useEffect, useCallback } from "react";
import { PredictionResponse } from "../types/schema";

const STORAGE_KEY = "agrosense_history";

export interface HistoryItem {
  id: string;
  timestamp: number;
  input: Record<string, number>;
  result: PredictionResponse;
}

export function useHistory() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setHistory(JSON.parse(stored));
      } catch (e) {
        console.error("Failed to parse history", e);
      }
    }
    setIsLoaded(true);
  }, []);

  const addToHistory = useCallback((input: Record<string, number>, result: PredictionResponse) => {
    const newItem: HistoryItem = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: Date.now(),
      input,
      // Sort predictions by confidence descending without mutating original result
      result: {
        ...result,
        predictions: [...result.predictions].sort((a, b) => b.confidence - a.confidence)
      }
    };

    setHistory(prev => {
      const updated = [newItem, ...prev].slice(0, 50); // Keep last 50
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
    
    return newItem.id;
  }, []);

  const clearHistory = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setHistory([]);
  }, []);

  const getHistoryItem = useCallback((id: string) => {
    return history.find(item => item.id === id);
  }, [history]);

  return {
    history,
    isLoaded,
    addToHistory,
    clearHistory,
    getHistoryItem
  };
}
