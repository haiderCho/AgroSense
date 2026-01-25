import { useMutation } from "@tanstack/react-query";
import { PredictionFormData, PredictionResponse } from "../types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/predict";

export const usePredictCrop = () => {
  return useMutation({
    mutationFn: async (data: PredictionFormData): Promise<PredictionResponse> => {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`API Connection Failed: ${response.statusText}`);
      }

      return response.json();
    },
    onError: (error) => {
        console.error("Prediction API Error:", error);
    }
  });
};
