"use client";

import {
  AlertTriangle, CheckCircle2, TrendingDown, TrendingUp,
  Info, Activity, Eye,
} from "lucide-react";
import { AVRGauge } from "./AVRGauge";
import { ImageViewer } from "./ImageViewer";
import type { AnalysisResponse } from "@/types";

interface ResultsPanelProps {
  data: AnalysisResponse;
}

function MetricBox({ label, value, unit, sub }: {
  label: string; value: string | number | null; unit?: string; sub?: string;
}) {
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
      <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-bold text-slate-900 tabular-nums">
          {value !== null && value !== undefined ? value : "—"}
        </span>
        {unit && <span className="text-xs text-slate-400">{unit}</span>}
      </div>
      {sub && <p className="text-[10px] text-slate-400 mt-0.5">{sub}</p>}
    </div>
  );
}

function FlagBanner({ flag }: { flag: string }) {
  const map: Record<string, { icon: React.ReactNode; title: string; bg: string; border: string; text: string; }> = {
    normal: {
      icon: <CheckCircle2 size={14} />,
      title: "Normal AVR — healthy microvascular calibre",
      bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700",
    },
    low_avr: {
      icon: <TrendingDown size={14} />,
      title: "Low AVR — possible arteriolar narrowing",
      bg: "bg-red-50", border: "border-red-200", text: "text-red-700",
    },
    high_avr: {
      icon: <TrendingUp size={14} />,
      title: "High AVR — possible venular dilation",
      bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-700",
    },
    insufficient_arterioles: {
      icon: <AlertTriangle size={14} />,
      title: "Insufficient arterioles for AVR",
      bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700",
    },
    insufficient_venules: {
      icon: <AlertTriangle size={14} />,
      title: "Insufficient venules for AVR",
      bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700",
    },
    insufficient_segments: {
      icon: <AlertTriangle size={14} />,
      title: "Insufficient vessel segments in measurement zone",
      bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700",
    },
    zero_crve: {
      icon: <Info size={14} />,
      title: "CRVE is zero — AVR undefined",
      bg: "bg-slate-50", border: "border-slate-200", text: "text-slate-600",
    },
    unknown: {
      icon: <Info size={14} />,
      title: "Analysis result unavailable",
      bg: "bg-slate-50", border: "border-slate-200", text: "text-slate-600",
    },
  };

  const c = map[flag] ?? map.unknown;
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-semibold ${c.bg} ${c.border} ${c.text}`}>
      {c.icon}
      {c.title}
    </div>
  );
}

export function ResultsPanel({ data }: ResultsPanelProps) {
  const { result, images } = data;

  return (
    <div className="space-y-4">

      {/* AVR Gauge card */}
      <div className="card p-6">
        <div className="card-header">
          <div className="w-7 h-7 rounded-lg bg-sky-50 border border-sky-200 flex items-center justify-center">
            <Activity size={14} className="text-sky-600" />
          </div>
          <h2 className="text-sm font-semibold text-slate-800">AVR Result</h2>
          <span className="ml-auto text-xs text-slate-400">Knudtson 2003</span>
        </div>

        <div className="flex justify-center mb-4">
          <AVRGauge value={result.AVR} flag={result.flag} />
        </div>

        <div className="grid grid-cols-2 gap-3 mb-3">
          <MetricBox label="CRAE" value={result.CRAE?.toFixed(3) ?? null} unit="px" sub="Arteriole equivalent" />
          <MetricBox label="CRVE" value={result.CRVE?.toFixed(3) ?? null} unit="px" sub="Venule equivalent" />
        </div>

        <FlagBanner flag={result.flag} />
      </div>

      {/* Clinical interpretation */}
      {result.interpretation && (
        <div className="card p-4">
          <div className="flex items-start gap-2.5">
            <div className="w-6 h-6 rounded-md bg-slate-100 border border-slate-200 flex items-center justify-center flex-shrink-0 mt-0.5">
              <Eye size={12} className="text-slate-500" />
            </div>
            <div>
              <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                Clinical Interpretation
              </p>
              <p className="text-sm text-slate-600 leading-relaxed">{result.interpretation}</p>
            </div>
          </div>
        </div>
      )}

      {/* Segment statistics */}
      {result.segments && (
        <div className="card p-4">
          <p className="label-text mb-3">Segment Statistics</p>
          <div className="grid grid-cols-3 gap-2">
            {[
              { v: result.segments.total,      label: "Total",      color: "text-slate-700" },
              { v: result.segments.arterioles,  label: "Arterioles", color: "text-red-600"   },
              { v: result.segments.venules,     label: "Venules",    color: "text-blue-600"  },
              { v: result.segments.in_zone,     label: "In Zone",    color: "text-sky-600"   },
              { v: result.segments.uncertain,   label: "Uncertain",  color: "text-amber-600" },
              { v: result.segments.rejected,    label: "Rejected",   color: "text-slate-400" },
            ].map(({ v, label, color }) => (
              <div key={label} className="bg-slate-50 border border-slate-100 rounded-xl p-2.5 text-center">
                <div className={`text-lg font-bold tabular-nums ${color}`}>{v}</div>
                <div className="text-[10px] text-slate-400 font-medium mt-0.5">{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disc detection */}
      {result.disc?.confidence !== null && (
        <div className="card p-4">
          <p className="label-text mb-2">Optic Disc Detection</p>
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs">
            {[
              { label: "Centre", val: `(${result.disc.cx}, ${result.disc.cy})` },
              { label: "Radius", val: `${result.disc.r} px` },
              { label: "Confidence", val: `${((result.disc.confidence ?? 0) * 100).toFixed(1)}%` },
              { label: "Method", val: result.disc.method?.replace("_", " ") ?? "—" },
            ].map(({ label, val }) => (
              <div key={label}>
                <span className="text-slate-400">{label}: </span>
                <span className="text-slate-700 font-medium capitalize">{val}</span>
              </div>
            ))}
            {result.disc.flag && (
              <div className="flex items-center gap-1 text-amber-600 font-medium w-full mt-0.5">
                <AlertTriangle size={10} />
                {result.disc.flag.replace(/_/g, " ")}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Pipeline visualisations */}
      {Object.keys(images).length > 0 && (
        <div className="card p-4">
          <p className="label-text mb-3">Pipeline Visualisations</p>
          <div style={{ height: "280px" }}>
            <ImageViewer images={images} />
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg bg-slate-50 border border-slate-200">
        <AlertTriangle size={12} className="text-slate-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-slate-400 leading-relaxed">
          <strong className="text-slate-500">Research use only.</strong>{" "}
          Pixel-based widths, not microns. Not validated for clinical diagnosis.
        </p>
      </div>
    </div>
  );
}
