"use client";

import { useState, useEffect } from "react";
import { PredictionResponse } from "../types";

const HISTORY_KEY = "agrosense_history";
const MAX_HISTORY = 5;

export interface HistoryItem {
  id: string;
  timestamp: number;
  result: PredictionResponse;
  label: string; // e.g., "Rice (98%)"
}

export function useHistory() {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(HISTORY_KEY);
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history", e);
      }
    }
  }, []);

  const addToHistory = (result: PredictionResponse) => {
    const topPrediction = result.predictions.sort((a, b) => b.confidence - a.confidence)[0];
    const id = crypto.randomUUID();
    const newItem: HistoryItem = {
      id,
      timestamp: Date.now(),
      result,
      label: `${topPrediction.crop} (${Math.round(topPrediction.confidence * 100)}%)`
    };

    setHistory((prev) => {
      const newHistory = [newItem, ...prev].slice(0, MAX_HISTORY);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
      return newHistory;
    });

    return id;
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
  };

  return { history, addToHistory, clearHistory };
}
