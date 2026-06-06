"use client";

import { useState } from "react";
import { ZoomIn, ZoomOut, RotateCcw, ImageOff } from "lucide-react";
import type { AnalysisImages } from "@/types";

interface Tab { key: keyof AnalysisImages; label: string; description: string; dot: string; }

const TABS: Tab[] = [
  { key: "original",   label: "Original", description: "Uploaded fundus image",        dot: "bg-slate-400" },
  { key: "overlay",    label: "Vessels",  description: "Segmented vessel overlay",      dot: "bg-emerald-500" },
  { key: "disc",       label: "Disc",     description: "Optic disc detection",          dot: "bg-yellow-500" },
  { key: "zone",       label: "Zone",     description: "1r–2r measurement zone",        dot: "bg-blue-500" },
  { key: "classified", label: "A/V Map",  description: "Arteriole/venule classification", dot: "bg-red-500" },
];

export function ImageViewer({ images }: { images: AnalysisImages }) {
  const [activeTab, setActiveTab] = useState<keyof AnalysisImages>("original");
  const [zoom, setZoom] = useState(1);

  const available = TABS.filter((t) => images[t.key]);
  const activeImage = images[activeTab];
  const activeMeta = TABS.find((t) => t.key === activeTab)!;

  if (!available.length) return null;

  return (
    <div className="flex flex-col h-full">
      {/* Tab row */}
      <div className="flex items-center gap-1 mb-2 overflow-x-auto pb-0.5 flex-shrink-0">
        {available.map((tab) => (
          <button
            key={tab.key}
            onClick={() => { setActiveTab(tab.key); setZoom(1); }}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all border ${
              activeTab === tab.key
                ? "bg-sky-700 border-sky-700 text-white"
                : "bg-white border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300"
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${activeTab === tab.key ? "bg-white/60" : tab.dot}`} />
            {tab.label}
          </button>
        ))}

        {/* Zoom controls */}
        <div className="ml-auto flex items-center gap-0.5 border border-slate-200 rounded-lg overflow-hidden bg-white">
          <button onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))} className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-colors" title="Zoom out">
            <ZoomOut size={13} />
          </button>
          <span className="text-[10px] text-slate-400 w-9 text-center font-medium">
            {Math.round(zoom * 100)}%
          </span>
          <button onClick={() => setZoom((z) => Math.min(3, z + 0.25))} className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-colors" title="Zoom in">
            <ZoomIn size={13} />
          </button>
          <button onClick={() => setZoom(1)} className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-colors border-l border-slate-200" title="Reset">
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      {/* Image area */}
      <div className="flex-1 relative rounded-xl overflow-hidden bg-slate-900 border border-slate-200 min-h-[160px]">
        {activeImage ? (
          <div className="absolute inset-0 overflow-auto flex items-center justify-center p-2">
            <img
              src={activeImage}
              alt={activeMeta.description}
              style={{
                transform: `scale(${zoom})`,
                transformOrigin: "center",
                transition: "transform 0.15s ease",
                maxWidth: zoom > 1 ? "none" : "100%",
                maxHeight: zoom > 1 ? "none" : "100%",
                objectFit: "contain",
              }}
              className="rounded"
            />
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-600">
            <ImageOff size={28} />
            <span className="text-xs mt-2">Not available</span>
          </div>
        )}

        {/* Overlay label */}
        <div className="absolute bottom-2 left-2 pointer-events-none">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-black/50 backdrop-blur-sm">
            <span className={`w-1.5 h-1.5 rounded-full ${activeMeta.dot}`} />
            <span className="text-[10px] text-white font-medium">{activeMeta.description}</span>
          </div>
        </div>
      </div>

      {/* Legend */}
      {activeTab === "classified" && (
        <div className="mt-1.5 flex items-center gap-3 text-[10px] text-slate-500">
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-sm bg-red-500 inline-block" /> Arterioles</span>
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-sm bg-blue-500 inline-block" /> Venules</span>
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-sm bg-yellow-400 inline-block" /> Uncertain</span>
        </div>
      )}
      {activeTab === "zone" && (
        <div className="mt-1.5 flex items-center gap-3 text-[10px] text-slate-500">
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-sm bg-emerald-500 inline-block" /> Kept</span>
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-sm bg-red-500 inline-block" /> Rejected</span>
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-sm bg-blue-400 inline-block" /> 1r/2r rings</span>
        </div>
      )}
    </div>
  );
}
