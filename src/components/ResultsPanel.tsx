"use client";

import {
  AlertTriangle, CheckCircle2, TrendingDown, TrendingUp,
  Info, Activity, Eye,
} from "lucide-react";
import { AVRGauge } from "./AVRGauge";
import { ImageViewer } from "./ImageViewer";
import type { AnalysisResponse, VesselAudit, Reliability } from "@/types";

interface ResultsPanelProps {
  data: AnalysisResponse;
}

/** Honest confidence banner — automated A/V classification is noisy. */
function ReliabilityBanner({ reliability }: { reliability: Reliability }) {
  if (reliability.level === "n/a") return null;
  const styles: Record<string, { bg: string; border: string; text: string; label: string }> = {
    high:     { bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700", label: "High confidence" },
    moderate: { bg: "bg-amber-50",   border: "border-amber-200",   text: "text-amber-700",   label: "Moderate confidence — interpret with care" },
    low:      { bg: "bg-red-50",     border: "border-red-200",     text: "text-red-700",     label: "Low confidence — treat as a rough estimate" },
  };
  const c = styles[reliability.level] ?? styles.low;
  return (
    <div className={`mt-3 rounded-lg border px-3 py-2 ${c.bg} ${c.border}`}>
      <div className="flex items-center gap-2">
        {reliability.level === "high"
          ? <CheckCircle2 size={13} className={c.text} />
          : <AlertTriangle size={13} className={c.text} />}
        <span className={`text-xs font-semibold ${c.text}`}>
          {c.label}
        </span>
        <span className={`ml-auto text-[10px] font-medium ${c.text} opacity-70`}>
          {reliability.score}/100
        </span>
      </div>
      {reliability.level !== "high" && reliability.reasons.length > 0 && (
        <ul className="mt-1.5 space-y-0.5 pl-5 list-disc">
          {reliability.reasons.map((r, i) => (
            <li key={i} className={`text-[11px] leading-snug ${c.text} opacity-90`}>{r}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * Shows the journey of every detected vessel: accepted (green, fed into the
 * AVR) vs rejected at each stage (with the reason). Stages are sequential.
 */
function VesselFunnel({ audit }: { audit: VesselAudit }) {
  const rows: Array<{ label: string; n: number; kind: "accept" | "reject"; hint: string }> = [
    { label: "Detected vessel segments", n: audit.detected, kind: "accept", hint: "found in Step 1" },
    { label: "Rejected — outside measurement zone", n: audit.rejected_outside_zone, kind: "reject", hint: "not in the 1r–2r ring around the disc" },
    { label: "Rejected — colour too ambiguous", n: audit.rejected_uncertain, kind: "reject", hint: "could not call arteriole vs venule" },
    { label: "Rejected — abnormal width (artefact)", n: audit.rejected_width_outlier, kind: "reject", hint: "likely merged vessels near the disc" },
    { label: "Accepted — used for AVR", n: audit.used_total, kind: "accept", hint: `${audit.used_arterioles} arterioles · ${audit.used_venules} venules` },
  ];

  return (
    <div className="card p-4">
      <p className="label-text mb-3">Vessel Acceptance Breakdown</p>
      <div className="space-y-1.5">
        {rows.map((r) => (
          <div
            key={r.label}
            className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${
              r.kind === "accept"
                ? "bg-emerald-50 border-emerald-200"
                : "bg-slate-50 border-slate-200"
            }`}
          >
            <div
              className={`text-base font-bold tabular-nums w-8 text-center ${
                r.kind === "accept" ? "text-emerald-700" : "text-slate-400"
              }`}
            >
              {r.n}
            </div>
            <div className="min-w-0">
              <div
                className={`text-xs font-semibold ${
                  r.kind === "accept" ? "text-emerald-800" : "text-slate-600"
                }`}
              >
                {r.label}
              </div>
              <div className="text-[10px] text-slate-400">{r.hint}</div>
            </div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-400 mt-2.5 leading-relaxed">
        Of <strong>{audit.detected}</strong> detected, <strong>{audit.in_zone}</strong> fell
        in the measurement zone; <strong>{audit.classified_arterioles + audit.classified_venules}</strong>{" "}
        were classified and <strong>{audit.used_total}</strong> survived quality checks to
        compute the AVR. Accepted vessels are highlighted green in the zone &amp; classification overlays below.
      </p>
    </div>
  );
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

        {result.reliability && <ReliabilityBanner reliability={result.reliability} />}
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

      {/* Vessel accept/reject funnel */}
      {result.vessel_audit && (
        <VesselFunnel audit={result.vessel_audit} />
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
