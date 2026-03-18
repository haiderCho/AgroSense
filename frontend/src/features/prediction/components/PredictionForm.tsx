"use client";

import React, { useCallback } from "react";
import { useForm, type Resolver } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Sprout, Beaker, Thermometer, Droplets, Wind, TestTube, Loader2 } from "lucide-react";
import { PredictionFormSchema, type PredictionFormData, type PredictionResponse } from "../types/schema";
import { usePredictCrop } from "../api/usePredictCrop";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "@/components/ui/toast";
import { cn } from "@/lib/utils";

interface PredictionFormProps {
  onSuccess: (data: PredictionFormData, result: PredictionResponse) => void;
}

const formItems = [
  { id: "N", label: "Nitrogen", icon: <Beaker className="w-4 h-4" />, placeholder: "e.g. 90", unit: "mg/kg" },
  { id: "P", label: "Phosphorus", icon: <TestTube className="w-4 h-4" />, placeholder: "e.g. 42", unit: "mg/kg" },
  { id: "K", label: "Potassium", icon: <Wind className="w-4 h-4" />, placeholder: "e.g. 43", unit: "mg/kg" },
  { id: "temperature", label: "Temperature", icon: <Thermometer className="w-4 h-4" />, placeholder: "e.g. 25", unit: "°C" },
  { id: "humidity", label: "Humidity", icon: <Droplets className="w-4 h-4" />, placeholder: "e.g. 70", unit: "%" },
  { id: "ph", label: "pH Level", icon: <Sprout className="w-4 h-4" />, placeholder: "e.g. 6.5", unit: "0-14" },
  { id: "rainfall", label: "Rainfall", icon: <Droplets className="w-4 h-4" />, placeholder: "e.g. 100", unit: "mm" },
] as const;

export const PredictionForm = React.memo(({ onSuccess }: PredictionFormProps) => {
  const { mutate, isPending } = usePredictCrop();
  
  const form = useForm<PredictionFormData>({
    resolver: zodResolver(PredictionFormSchema) as unknown as Resolver<PredictionFormData>,
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

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = form;

  const onSubmit = useCallback(async (data: Record<string, number>) => {
    mutate(data as PredictionFormData, {
      onSuccess: (result: PredictionResponse) => {
        onSuccess(data as PredictionFormData, result);
        toast("Analysis complete!", "success");
      },
      onError: (err: Error) => {
        console.error(err);
        toast("Analysis failed. Please try again.", "error");
      },
    });
  }, [mutate, onSuccess]);

  return (
    <Card className="w-full border-border shadow-sm rounded-lg overflow-hidden">
      <CardHeader className="p-5 pb-4">
        <CardTitle className="text-xl font-bold flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <Sprout className="w-5 h-5 text-primary" />
            </div>
            Soil Diagnostic Input
        </CardTitle>
        <CardDescription className="text-sm pt-0.5 opacity-80">
          Provide accurate soil and environmental metrics for optimal crop recommendation.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {formItems.map((field) => (
              <div key={field.id} className="space-y-1.5">
                <div className="flex items-center justify-between px-0.5">
                  <Label 
                    htmlFor={field.id} 
                    className="flex items-center gap-1.5 text-xs font-bold text-muted-foreground uppercase tracking-wider"
                  >
                    {field.icon}
                    {field.label}
                  </Label>
                  <span className="text-[10px] text-muted-foreground/60 font-mono font-bold">
                    {field.unit}
                  </span>
                </div>
                <div className="relative">
                  <Input
                    id={field.id}
                    type="number"
                    step={field.id === "ph" || field.id === "temperature" || field.id === "rainfall" ? "0.1" : "1"}
                    placeholder={field.placeholder}
                    {...register(field.id as keyof PredictionFormData)}
                    className={cn(
                      "h-10 text-sm px-3 rounded-lg bg-background/50 focus-visible:ring-primary/20 transition-all",
                      errors[field.id as keyof PredictionFormData] ? "border-destructive focus-visible:ring-destructive" : ""
                    )}
                    aria-invalid={!!errors[field.id as keyof PredictionFormData]}
                  />
                  {errors[field.id as keyof PredictionFormData] && (
                    <p className="text-[10px] text-destructive mt-1 font-bold">
                      {errors[field.id as keyof PredictionFormData]?.message}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="pt-6 border-t">
            <Button 
                type="submit" 
                className="w-full h-12 text-base font-bold rounded-lg shadow-sm hover:shadow-md transition-all active:scale-[0.98]"
                disabled={isPending}
            >
              {isPending ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Processing...</span>
                </div>
              ) : (
                "Run Soil Analysis"
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
});

PredictionForm.displayName = "PredictionForm";
