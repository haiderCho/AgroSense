"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import { PredictionFormData, PredictionResponse } from "../types";
import { usePredictCrop } from "../api/usePredictCrop";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { RotateCcw, Activity, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/toast";
import { debounce } from "lodash";
import { cn } from "@/lib/utils";

interface SimulationLabProps {
  initialData?: PredictionFormData;
}

export function SimulationLab({ initialData }: SimulationLabProps) {
  const defaults: PredictionFormData = useMemo(() => initialData || {
    N: 90,
    P: 42,
    K: 43,
    ph: 6.5,
    temperature: 20.8,
    humidity: 82,
    rainfall: 202,
  }, [initialData]);

  const [params, setParams] = useState<PredictionFormData>(defaults);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  
  const { mutate: predict, isPending } = usePredictCrop();

  const debouncedPredict = useMemo(
    () => debounce((data: PredictionFormData) => {
      predict(data, {
        onSuccess: (result) => setPrediction(result),
      });
    }, 500),
    [predict]
  );

  useEffect(() => {
    debouncedPredict(params);
    return () => {
      debouncedPredict.cancel();
    };
  }, [params, debouncedPredict]);

  const handleSliderChange = (name: keyof PredictionFormData, value: number[]) => {
    setParams(prev => ({ ...prev, [name]: value[0] }));
  };

  const resetParams = useCallback(() => {
    setParams(defaults);
    predict(defaults, {
        onSuccess: (result) => setPrediction(result)
    });
    toast("Simulation reset", "info");
  }, [defaults, predict]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
      <Card className="lg:col-span-2 border-border bg-card shadow-md">
        <CardHeader className="flex flex-row items-center justify-between border-b border-border pb-6 mb-6 bg-muted/5">
          <div className="space-y-1">
            <CardTitle className="text-2xl font-bold flex items-center gap-3">
                <FlaskConical className="w-6 h-6 text-primary" />
                Simulation Lab
            </CardTitle>
            <CardDescription className="text-base">Experiment with soil & climate parameters in real-time.</CardDescription>
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={resetParams}
            className="h-10 w-10 transition-all hover:rotate-[-90deg]"
            title="Reset to defaults"
          >
            <RotateCcw className="w-5 h-5" />
          </Button>
        </CardHeader>
        <CardContent className="p-8 pt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-10 gap-y-8">
            {/* N Slider */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-semibold text-foreground uppercase tracking-wider">Nitrogen (N) <span className="text-muted-foreground font-normal lowercase">(mg/kg)</span></Label>
                <div className="text-sm font-black text-primary bg-primary/5 px-2 py-0.5 rounded border border-primary/10">{params.N}</div>
              </div>
              <Slider 
                value={[params.N]} 
                min={0} max={140} step={1} 
                onValueChange={(v) => handleSliderChange("N", v)} 
              />
            </div>

            {/* P Slider */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-semibold text-foreground uppercase tracking-wider">Phosphorus (P) <span className="text-muted-foreground font-normal lowercase">(mg/kg)</span></Label>
                <div className="text-sm font-black text-primary bg-primary/5 px-2 py-0.5 rounded border border-primary/10">{params.P}</div>
              </div>
              <Slider 
                value={[params.P]} 
                min={5} max={145} step={1} 
                onValueChange={(v) => handleSliderChange("P", v)} 
              />
            </div>

            {/* K Slider */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-semibold text-foreground uppercase tracking-wider">Potassium (K) <span className="text-muted-foreground font-normal lowercase">(mg/kg)</span></Label>
                <div className="text-sm font-black text-primary bg-primary/5 px-2 py-0.5 rounded border border-primary/10">{params.K}</div>
              </div>
              <Slider 
                value={[params.K]} 
                min={5} max={205} step={1} 
                onValueChange={(v) => handleSliderChange("K", v)} 
              />
            </div>

            {/* pH Slider */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-semibold text-foreground uppercase tracking-wider">pH Level <span className="text-muted-foreground font-normal lowercase">(0-14)</span></Label>
                <div className="text-sm font-black text-primary bg-primary/5 px-2 py-0.5 rounded border border-primary/10">{params.ph}</div>
              </div>
              <Slider 
                value={[params.ph]} 
                min={0} max={14} step={0.1} 
                onValueChange={(v) => handleSliderChange("ph", v)} 
              />
            </div>

            {/* Temp Slider */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-semibold text-foreground uppercase tracking-wider">Temperature <span className="text-muted-foreground font-normal lowercase">(°C)</span></Label>
                <div className="text-sm font-black text-primary bg-primary/5 px-2 py-0.5 rounded border border-primary/10">{params.temperature}</div>
              </div>
              <Slider 
                value={[params.temperature]} 
                min={5} max={50} step={0.1} 
                onValueChange={(v) => handleSliderChange("temperature", v)} 
              />
            </div>

            {/* Humidity Slider */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-semibold text-foreground uppercase tracking-wider">Humidity <span className="text-muted-foreground font-normal lowercase">(%)</span></Label>
                <div className="text-sm font-black text-primary bg-primary/5 px-2 py-0.5 rounded border border-primary/10">{params.humidity}</div>
              </div>
              <Slider 
                value={[params.humidity]} 
                min={10} max={100} step={1} 
                onValueChange={(v) => handleSliderChange("humidity", v)} 
              />
            </div>

            {/* Rainfall Slider */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-semibold text-foreground uppercase tracking-wider">Rainfall <span className="text-muted-foreground font-normal lowercase">(mm)</span></Label>
                <div className="text-sm font-black text-primary bg-primary/5 px-2 py-0.5 rounded border border-primary/10">{params.rainfall}</div>
              </div>
              <Slider 
                value={[params.rainfall]} 
                min={20} max={300} step={1} 
                onValueChange={(v) => handleSliderChange("rainfall", v)} 
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-1 border-border bg-card shadow-md flex flex-col items-center justify-center p-10 relative overflow-hidden">
        <div className="absolute top-0 right-0 p-4">
          <Activity className={cn("w-5 h-5", isPending ? "text-primary animate-pulse" : "text-muted-foreground/20")} />
        </div>
        
        {isPending && !prediction && (
          <div className="flex flex-col items-center gap-4">
             <div className="w-10 h-10 border-4 border-primary/20 border-t-primary rounded-full animate-spin" />
             <p className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Analysing...</p>
          </div>
        )}
 
        {prediction ? (
          <div className="text-center space-y-8 w-full animate-in zoom-in-95 duration-300">
            <div className="space-y-2">
              <p className="text-xs font-black uppercase tracking-widest text-muted-foreground/60">SIMULATED MATCH</p>
              <h3 className="text-4xl font-black text-primary capitalize tracking-tighter">
                {prediction.consensus_crop}
              </h3>
            </div>
            
            <div className="space-y-3 px-4">
              <div className="flex justify-between items-center text-xs font-black text-muted-foreground uppercase tracking-tighter">
                <span>Confidence Score</span>
                <span className="text-primary">
                  {Math.round((prediction.predictions?.[0]?.confidence || 0) * 100)}%
                </span>
              </div>
              <div className="h-3 w-full bg-muted rounded-full overflow-hidden border border-border">
                <div 
                  className="h-full bg-primary transition-all duration-700 ease-out"
                  style={{ width: `${(prediction.predictions?.[0]?.confidence || 0) * 100}%` }}
                />
              </div>
            </div>
          </div>
        ) : !isPending && (
          <div className="text-center space-y-6">
             <div className="w-16 h-16 rounded-full bg-muted/30 flex items-center justify-center mx-auto">
                <FlaskConical className="w-8 h-8 text-muted-foreground/40" />
             </div>
             <p className="text-base font-semibold text-muted-foreground max-w-[180px] mx-auto">Adjust parameters to simulate crop yield</p>
          </div>
        )}
      </Card>
    </div>
  );
}
