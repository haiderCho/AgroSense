"use client";

import { useForm, type Resolver } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { PredictionFormSchema, type PredictionFormData, type PredictionResponse } from "../types";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Loader2, Sprout } from "lucide-react";
import { usePredictCrop } from "../api/usePredictCrop";

interface PredictionFormProps {
  onAnalysisComplete: (result: PredictionResponse) => void;
}

export function PredictionForm({ onAnalysisComplete }: PredictionFormProps) {
  const { mutate: predict, isPending } = usePredictCrop();

  const form = useForm<PredictionFormData>({
    resolver: zodResolver(PredictionFormSchema) as Resolver<PredictionFormData>,
    defaultValues: {
      N: 90,
      P: 42,
      K: 43,
      temperature: 20.8,
      humidity: 82,
      ph: 6.5,
      rainfall: 202,
    },
  });

  function onSubmit(data: PredictionFormData) {
    predict(data, {
      onSuccess: (result) => {
         onAnalysisComplete(result);
      },
      onError: () => {
         alert("Failed to connect to AgroSense API. Ensure backend is running on port 8000.");
      }
    });
  }

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto backdrop-blur-sm bg-card/95 border-primary/20 shadow-2xl">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sprout className="w-6 h-6 text-primary" />
          <span>Soil Analysis & Crop Prediction</span>
        </CardTitle>
        <CardDescription>
          Enter your soil parameters below. Our multi-model AI ensemble will analyze suitability.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <motion.div 
            variants={container}
            initial="hidden"
            animate="show"
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {/* Nitrogen */}
            <motion.div variants={item} className="space-y-2">
              <Label htmlFor="N">Nitrogen (N)</Label>
              <Input
                id="N"
                type="number"
                {...form.register("N")}
                className="font-mono"
              />
              {form.formState.errors.N && (
                <p className="text-xs text-destructive">{form.formState.errors.N.message}</p>
              )}
            </motion.div>

            {/* Phosphorus */}
            <motion.div variants={item} className="space-y-2">
              <Label htmlFor="P">Phosphorus (P)</Label>
              <Input
                id="P"
                type="number"
                {...form.register("P")}
                className="font-mono"
              />
              {form.formState.errors.P && (
                <p className="text-xs text-destructive">{form.formState.errors.P.message}</p>
              )}
            </motion.div>

            {/* Potassium */}
            <motion.div variants={item} className="space-y-2">
              <Label htmlFor="K">Potassium (K)</Label>
              <Input
                id="K"
                type="number"
                {...form.register("K")}
                className="font-mono"
              />
              {form.formState.errors.K && (
                <p className="text-xs text-destructive">{form.formState.errors.K.message}</p>
              )}
            </motion.div>
            
             {/* pH Level */}
             <motion.div variants={item} className="space-y-2">
              <Label htmlFor="ph">pH Level</Label>
              <Input
                id="ph"
                type="number"
                step="0.1"
                {...form.register("ph")}
                className="font-mono"
              />
              {form.formState.errors.ph && (
                <p className="text-xs text-destructive">{form.formState.errors.ph.message}</p>
              )}
            </motion.div>

            {/* Temperature */}
            <motion.div variants={item} className="space-y-2">
              <Label htmlFor="temperature">Temperature (°C)</Label>
              <Input
                id="temperature"
                type="number"
                step="0.1"
                {...form.register("temperature")}
                className="font-mono"
              />
              {form.formState.errors.temperature && (
                <p className="text-xs text-destructive">{form.formState.errors.temperature.message}</p>
              )}
            </motion.div>

            {/* Humidity */}
            <motion.div variants={item} className="space-y-2">
              <Label htmlFor="humidity">Humidity (%)</Label>
              <Input
                id="humidity"
                type="number"
                {...form.register("humidity")}
                className="font-mono"
              />
              {form.formState.errors.humidity && (
                <p className="text-xs text-destructive">{form.formState.errors.humidity.message}</p>
              )}
            </motion.div>

             {/* Rainfall */}
             <motion.div variants={item} className="md:col-span-2 space-y-2">
              <Label htmlFor="rainfall">Rainfall (mm)</Label>
              <Input
                id="rainfall"
                type="number"
                step="0.1"
                {...form.register("rainfall")}
                className="font-mono"
              />
              {form.formState.errors.rainfall && (
                <p className="text-xs text-destructive">{form.formState.errors.rainfall.message}</p>
              )}
            </motion.div>

          </motion.div>

          <motion.div
             initial={{ opacity: 0 }}
             animate={{ opacity: 1 }}
             transition={{ delay: 0.8 }}
             className="pt-4"
          >
            <Button 
                type="submit" 
                className="w-full h-12 text-lg font-bold tracking-wide"
                disabled={isPending}
            >
              {isPending ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  ANALYZING DATA...
                </>
              ) : (
                "RUN PREDICTION MODEL"
              )}
            </Button>
          </motion.div>
        </form>
      </CardContent>
    </Card>
  );
}
