import { z } from "zod";

export const PredictionFormSchema = z.object({
  N: z.coerce.number().min(0).max(140, "Nitrogen (N) must be between 0 and 140"),
  P: z.coerce.number().min(5).max(145, "Phosphorus (P) must be between 5 and 145"),
  K: z.coerce.number().min(5).max(205, "Potassium (K) must be between 5 and 205"),
  temperature: z.coerce.number().min(5).max(50, "Temperature must be between 5°C and 50°C"),
  humidity: z.coerce.number().min(10).max(100, "Humidity must be between 10% and 100%"),
  ph: z.coerce.number().min(0).max(14, "pH must be between 0 and 14"),
  rainfall: z.coerce.number().min(20).max(300, "Rainfall must be between 20mm and 300mm"),
});

export type PredictionFormData = z.infer<typeof PredictionFormSchema>;

export interface PredictionResponse {
  predictions: {
    model: string;
    crop: string;
    confidence: number;
    explanation?: Record<string, number>; // Feature importance
  }[];
  consensus_crop: string;
}
