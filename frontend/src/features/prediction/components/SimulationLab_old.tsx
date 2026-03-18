"use client";

import { useState, useCallback, useEffect } from "react";
import { PredictionFormData, PredictionResponse } from "../types";
import { usePredictCrop } from "../api/usePredictCrop";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { RotateCcw, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { debounce } from "lodash"; 
// Note: You might need to install lodash or write a simple debounce if not present. 
// I'll write a simple debounce inside useEffect to avoid dependencies if possible, 
// or assume lodash is available. Let's use a custom hook approach for simplicity/dependency-free.

interface SimulationLabProps {
  initialData?: PredictionFormData;
}

export function SimulationLab({ initialData }: SimulationLabProps) {
  // distinct defaults if nothing provided
  const defaults: PredictionFormData = initialData || {
    N: 90,
    P: 42,
    K: 43,
    temperature: 20.8,
    humidity: 82,
    ph: 6.5,
    rainfall: 202,
  };

  const [params, setParams] = useState<PredictionFormData>(defaults);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  
  const { mutate: predict, isPending } = usePredictCrop();

  // Debounced prediction function
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const debouncedPredict = useCallback(
    debounce((data: PredictionFormData) => {
      predict(data, {
        onSuccess: (result) => setPrediction(result),
      });
    }, 500),
    []
  );

  useEffect(() => {
    debouncedPredict(params);
    // Cleanup debounce on unmount
    return () => {
      debouncedPredict.cancel();
    };
  }, [params, debouncedPredict]);

  const handleSliderChange = (key: keyof PredictionFormData, value: number[]) => {
    setParams((prev) => ({ ...prev, [key]: value[0] }));
  };

  const resetParams = () => {
    setParams(defaults);
  };

  return (
    <Card className="border-primary/20 bg-gradient-to-br from-card to-background/50">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="text-xl flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              What-If Simulation Lab
            </CardTitle>
            <CardDescription>
              Adjust parameters to see how the AI adapts in real-time.
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={resetParams} className="h-8 gap-1">
            <RotateCcw className="w-3.5 h-3.5" /> Reset
          </Button>
        </div>
      </CardHeader>
      <CardContent className="grid lg:grid-cols-3 gap-8">
        
        {/* Controls Column */}
        <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6 p-1">
          {/* N Slider */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-xs font-mono text-muted-foreground">Nitrogen (N)</Label>
                <span className="text-xs font-mono font-bold">{params.N}</span>
             </div>
             <Slider 
                value={[params.N]} 
                min={0} max={140} step={1} 
                onValueChange={(v) => handleSliderChange("N", v)} 
                className="py-1"
             />
          </div>

          {/* P Slider */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-xs font-mono text-muted-foreground">Phosphorus (P)</Label>
                <span className="text-xs font-mono font-bold">{params.P}</span>
             </div>
             <Slider 
                value={[params.P]} 
                min={5} max={145} step={1} 
                onValueChange={(v) => handleSliderChange("P", v)} 
                className="py-1"
             />
          </div>

          {/* K Slider */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-xs font-mono text-muted-foreground">Potassium (K)</Label>
                <span className="text-xs font-mono font-bold">{params.K}</span>
             </div>
             <Slider 
                value={[params.K]} 
                min={5} max={205} step={1} 
                onValueChange={(v) => handleSliderChange("K", v)} 
                className="py-1"
             />
          </div>

          {/* Rainfall Slider */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-xs font-mono text-muted-foreground">Rainfall (mm)</Label>
                <span className="text-xs font-mono font-bold">{params.rainfall}</span>
             </div>
             <Slider 
                value={[params.rainfall]} 
                min={20} max={300} step={1} 
                onValueChange={(v) => handleSliderChange("rainfall", v)} 
                className="py-1"
             />
          </div>

          {/* pH Slider */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-xs font-mono text-muted-foreground">pH Level</Label>
                <span className="text-xs font-mono font-bold">{params.ph}</span>
             </div>
             <Slider 
                value={[params.ph]} 
                min={0} max={14} step={0.1} 
                onValueChange={(v) => handleSliderChange("ph", v)} 
                className="py-1"
             />
          </div>

          {/* Temp Slider */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-xs font-mono text-muted-foreground">Temperature (°C)</Label>
                <span className="text-xs font-mono font-bold">{params.temperature}</span>
             </div>
             <Slider 
                value={[params.temperature]} 
                min={5} max={50} step={0.1} 
                onValueChange={(v) => handleSliderChange("temperature", v)} 
                className="py-1"
             />
          </div>
        </div>

        {/* Live Result Column */}
        <div className="lg:col-span-1 bg-background/50 rounded-xl border border-border/50 p-6 flex flex-col items-center justify-center relative overflow-hidden transition-all">
          
          {isPending && !prediction && (
             <div className="absolute inset-0 bg-background/80 backdrop-blur-[1px] z-10 flex items-center justify-center">
                <div className="animate-pulse text-xs text-muted-foreground">Simulating...</div>
             </div>
          )}

          {prediction ? (
             <div className="text-center space-y-4 animate-in fade-in zoom-in-95 duration-300">
                <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-widest">
                   Projected Crop
                </h3>
                <div className="py-2">
                   <span className="text-4xl lg:text-5xl font-black text-primary capitalize drop-shadow-sm">
                      {prediction.consensus_crop}
                   </span>
                </div>
                
                {/* Mini Confidence Bar */}
                <div className="w-full max-w-[150px] space-y-1 mx-auto">
                    <div className="flex justify-between text-[10px] text-muted-foreground uppercase">
                       <span>Confidence</span>
                       <span>{Math.round(prediction.predictions[0]?.confidence * 100)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                       <div 
                          className="h-full bg-primary transition-all duration-500 ease-out"
                          style={{ width: `${prediction.predictions[0]?.confidence * 100}%` }}
                       />
                    </div>
                </div>
             </div>
          ) : (
             <div className="text-muted-foreground text-sm">Initializing simulation...</div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
