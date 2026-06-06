"use client";

import { Check, X, Loader2, Minus, Scan, Circle, Filter, Tag, Calculator } from "lucide-react";
import type { PipelineStep } from "@/types";

interface ProgressStepsProps {
  steps: PipelineStep[];
}

const STEP_ICONS = {
  step1: Scan,
  step2: Circle,
  step3: Filter,
  step4: Tag,
  step5: Calculator,
};

function StepIndicator({ step }: { step: PipelineStep }) {
  const Icon = STEP_ICONS[step.key];

  if (step.status === "running") {
    return (
      <div className="w-8 h-8 rounded-full bg-sky-700 border-2 border-sky-700 flex items-center justify-center flex-shrink-0">
        <Loader2 size={14} className="animate-spin text-white" />
      </div>
    );
  }
  if (step.status === "done") {
    return (
      <div className="w-8 h-8 rounded-full bg-emerald-500 border-2 border-emerald-500 flex items-center justify-center flex-shrink-0">
        <Check size={14} className="text-white" strokeWidth={2.5} />
      </div>
    );
  }
  if (step.status === "error") {
    return (
      <div className="w-8 h-8 rounded-full bg-red-500 border-2 border-red-500 flex items-center justify-center flex-shrink-0">
        <X size={14} className="text-white" strokeWidth={2.5} />
      </div>
    );
  }
  if (step.status === "skipped") {
    return (
      <div className="w-8 h-8 rounded-full bg-slate-200 border-2 border-slate-300 flex items-center justify-center flex-shrink-0">
        <Minus size={14} className="text-slate-400" />
      </div>
    );
  }
  // pending
  return (
    <div className="w-8 h-8 rounded-full bg-slate-100 border-2 border-slate-200 flex items-center justify-center flex-shrink-0">
      <Icon size={13} className="text-slate-400" />
    </div>
  );
}

function statusLabel(status: PipelineStep["status"]) {
  switch (status) {
    case "running": return <span className="text-xs text-sky-600 font-semibold step-running">Processing</span>;
    case "done":    return <span className="text-xs text-emerald-600 font-semibold">Complete</span>;
    case "error":   return <span className="text-xs text-red-600 font-semibold">Failed</span>;
    case "skipped": return <span className="text-xs text-slate-400 font-medium">Skipped</span>;
    default:        return <span className="text-xs text-slate-300 font-medium">Waiting</span>;
  }
}

function connectorColor(status: PipelineStep["status"]) {
  if (status === "done") return "bg-emerald-300";
  if (status === "error") return "bg-red-300";
  return "bg-slate-200";
}

export function ProgressSteps({ steps }: ProgressStepsProps) {
  return (
    <div>
      <p className="label-text mb-4">Analysis Pipeline</p>

      <div className="space-y-0">
        {steps.map((step, idx) => (
          <div key={step.key}>
            <div className={`flex items-center gap-3 py-2.5 px-3 rounded-xl transition-all duration-200 ${step.status === "running" ? "bg-sky-50 border border-sky-200" : step.status === "done" ? "bg-emerald-50/40" : "bg-transparent"}`}>
              <StepIndicator step={step} />

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className={`text-sm font-medium ${step.status === "running" ? "text-sky-800" : step.status === "done" ? "text-slate-700" : "text-slate-500"}`}>
                    {step.label}
                  </span>
                  {statusLabel(step.status)}
                </div>
                <p className="text-xs text-slate-400 mt-0.5 truncate">
                  {step.description}
                </p>
              </div>
            </div>

            {/* Connector */}
            {idx < steps.length - 1 && (
              <div className="ml-[19px] flex">
                <div className={`w-0.5 h-3 ${connectorColor(step.status)} transition-colors duration-300`} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
