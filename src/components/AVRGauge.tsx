"use client";

import { useEffect, useState } from "react";

interface AVRGaugeProps {
  value: number | null;
  flag: string;
  animated?: boolean;
}

const FLAG_CONFIG = {
  normal:      { color: "#059669", label: "Normal",   bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700" },
  low_avr:     { color: "#dc2626", label: "Low AVR",  bg: "bg-red-50",     border: "border-red-200",     text: "text-red-700"     },
  high_avr:    { color: "#d97706", label: "High AVR", bg: "bg-amber-50",   border: "border-amber-200",   text: "text-amber-700"   },
};

export function AVRGauge({ value, flag, animated = true }: AVRGaugeProps) {
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    if (value === null) return;
    if (!animated) { setDisplayed(value); return; }

    let start: number | null = null;
    const target = Math.min(Math.max(value, 0), 1.2);

    const step = (ts: number) => {
      if (!start) start = ts;
      const t = Math.min((ts - start) / 1200, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setDisplayed(ease * target);
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [value, animated]);

  const cfg = FLAG_CONFIG[flag as keyof typeof FLAG_CONFIG];

  // SVG geometry — half-circle gauge
  const cx = 110, cy = 100, r = 80, sw = 12;
  const clamp = Math.min(Math.max(displayed, 0), 1.2);
  const angleRad = Math.PI - (clamp / 1.2) * Math.PI;
  const nx = cx + r * Math.cos(Math.PI - angleRad);
  const ny = cy - r * Math.sin(angleRad);

  function arc(a0: number, a1: number) {
    const s = Math.PI - (a0 / 1.2) * Math.PI;
    const e = Math.PI - (a1 / 1.2) * Math.PI;
    const x1 = cx + r * Math.cos(s); const y1 = cy - r * Math.sin(s);
    const x2 = cx + r * Math.cos(e); const y2 = cy - r * Math.sin(e);
    return `M ${x1} ${y1} A ${r} ${r} 0 ${a1 - a0 > 0.6 ? 1 : 0} 1 ${x2} ${y2}`;
  }

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <svg width="220" height="125" viewBox="0 0 220 125">
          {/* Track */}
          <path d={arc(0, 1.2)} fill="none" stroke="#f1f5f9" strokeWidth={sw} strokeLinecap="round" />

          {/* Red zone 0–0.6 */}
          <path d={arc(0, 0.6)}   fill="none" stroke="#fca5a5" strokeWidth={sw - 2} strokeLinecap="butt" />
          {/* Green zone 0.6–0.8 */}
          <path d={arc(0.6, 0.8)} fill="none" stroke="#6ee7b7" strokeWidth={sw - 2} strokeLinecap="butt" />
          {/* Amber zone 0.8–1.2 */}
          <path d={arc(0.8, 1.2)} fill="none" stroke="#fcd34d" strokeWidth={sw - 2} strokeLinecap="butt" />

          {/* Zone boundary ticks */}
          {[0, 0.6, 0.8, 1.2].map((v) => {
            const a = Math.PI - (v / 1.2) * Math.PI;
            return (
              <line key={v}
                x1={cx + (r - 7) * Math.cos(a)} y1={cy - (r - 7) * Math.sin(a)}
                x2={cx + (r + 3) * Math.cos(a)} y2={cy - (r + 3) * Math.sin(a)}
                stroke="#94a3b8" strokeWidth="1"
              />
            );
          })}

          {/* Tick labels */}
          {[{ v: 0, lbl: "0" }, { v: 0.6, lbl: "0.6" }, { v: 0.8, lbl: "0.8" }, { v: 1.2, lbl: "1.2" }].map(({ v, lbl }) => {
            const a = Math.PI - (v / 1.2) * Math.PI;
            return (
              <text key={v}
                x={cx + (r + 17) * Math.cos(a)}
                y={cy - (r + 17) * Math.sin(a) + 4}
                fill="#94a3b8" fontSize="9" textAnchor="middle"
              >
                {lbl}
              </text>
            );
          })}

          {/* Needle */}
          {value !== null && (
            <>
              <line
                x1={cx} y1={cy} x2={nx} y2={ny}
                stroke={cfg?.color ?? "#64748b"} strokeWidth="2.5" strokeLinecap="round"
                style={{ transition: "x2 1.2s cubic-bezier(0.4,0,0.2,1), y2 1.2s cubic-bezier(0.4,0,0.2,1)" }}
              />
              <circle cx={cx} cy={cy} r={5} fill={cfg?.color ?? "#64748b"} />
              <circle cx={cx} cy={cy} r={2.5} fill="white" />
            </>
          )}
        </svg>

        {/* Value */}
        <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-center">
          {value !== null ? (
            <>
              <div className="text-2xl font-bold text-slate-900 leading-none tabular-nums">
                {value.toFixed(3)}
              </div>
              <div className="text-[10px] text-slate-400 font-medium mt-0.5 uppercase tracking-wider">
                AVR
              </div>
            </>
          ) : (
            <div className="text-slate-300 text-sm">—</div>
          )}
        </div>
      </div>

      {/* Flag badge */}
      {cfg && (
        <div className={`mt-2 px-3 py-1 rounded-md border text-xs font-semibold ${cfg.bg} ${cfg.border} ${cfg.text}`}>
          {cfg.label}
        </div>
      )}
    </div>
  );
}
