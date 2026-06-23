"use client";

import { useState, useCallback } from "react";
import { AlertCircle, FlaskConical } from "lucide-react";
import { Header } from "@/components/Header";
import { UploadZone } from "@/components/UploadZone";
import { ProgressSteps } from "@/components/ProgressSteps";
import { ResultsPanel } from "@/components/ResultsPanel";
import type { AnalysisResponse, PipelineStep } from "@/types";

const INITIAL_STEPS: PipelineStep[] = [
  {
    id: 1,
    key: "step1",
    label: "Vessel Segmentation",
    description: "Frangi vesselness filter + skeleton extraction",
    icon: "scan",
    status: "pending",
  },
  {
    id: 2,
    key: "step2",
    label: "Disc Detection",
    description: "Optic disc centre and radius localisation",
    icon: "circle",
    status: "pending",
  },
  {
    id: 3,
    key: "step3",
    label: "Zone Filtering",
    description: "Keep vessels in 1r–2r measurement annulus",
    icon: "filter",
    status: "pending",
  },
  {
    id: 4,
    key: "step4",
    label: "A/V Classification",
    description: "Colour + width score to label arterioles/venules",
    icon: "tag",
    status: "pending",
  },
  {
    id: 5,
    key: "step5",
    label: "AVR Calculation",
    description: "CRAE, CRVE and arteriovenous ratio (Knudtson 2003)",
    icon: "calculator",
    status: "pending",
  },
];

type AppState = "idle" | "analyzing" | "done" | "error";

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [appState, setAppState] = useState<AppState>("idle");
  const [steps, setSteps] = useState<PipelineStep[]>(INITIAL_STEPS);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const resetSteps = () =>
    setSteps(INITIAL_STEPS.map((s) => ({ ...s, status: "pending" })));

  const setStepStatus = (key: PipelineStep["key"], status: PipelineStep["status"]) =>
    setSteps((prev) => prev.map((s) => (s.key === key ? { ...s, status } : s)));

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setAppState("idle");
    setResult(null);
    setErrorMessage(null);
    resetSteps();
  }, []);

  const handleClear = () => {
    setSelectedFile(null);
    setAppState("idle");
    setResult(null);
    setErrorMessage(null);
    resetSteps();
  };

  const handleAnalyze = async () => {
    if (!selectedFile || appState === "analyzing") return;

    setAppState("analyzing");
    setResult(null);
    setErrorMessage(null);
    resetSteps();

    const formData = new FormData();
    formData.append("image", selectedFile);

    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
    setStepStatus("step1", "running");

    let responseData: AnalysisResponse | null = null;

    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "/api/analyze";
    const apiPromise = fetch(apiUrl, { method: "POST", body: formData })
      .then(async (r) => {
        const text = await r.text();
        try {
          return JSON.parse(text);
        } catch {
          // Server returned HTML/plain-text (e.g. a 500 error page or traceback)
          const snippet = text.replace(/<[^>]+>/g, " ").trim().slice(0, 300);
          return {
            status: "error",
            error: `Server error (HTTP ${r.status}) from ${apiUrl}: ${snippet || "empty response"}`,
          };
        }
      })
      .catch((err) => ({
        status: "error",
        error: `${String(err)} — API URL: ${apiUrl}. If this is a CORS or network error, make sure NEXT_PUBLIC_API_URL is set to your backend service URL in Render's environment settings and the frontend has been redeployed.`,
      }));

    // Track when the real API call finishes so the progress animation can
    // bail out immediately instead of running a fixed, padded duration.
    let apiDone = false;
    const apiTracked = apiPromise.then((d) => {
      responseData = d as AnalysisResponse;
      apiDone = true;
    });

    // Light-weight progress animation. Short per-step delays just so the user
    // sees movement; the loop exits the moment the backend responds, so a fast
    // analysis returns in well under a second rather than waiting ~6s.
    const progressSteps: Array<PipelineStep["key"]> = [
      "step1", "step2", "step3", "step4", "step5",
    ];
    const STEP_MS = 250;

    const advanceSteps = async () => {
      for (let i = 0; i < progressSteps.length; i++) {
        if (apiDone) break;                       // backend already finished
        setStepStatus(progressSteps[i], "running");
        await delay(STEP_MS);
        if (i < progressSteps.length - 1) setStepStatus(progressSteps[i], "done");
      }
    };

    await Promise.all([apiTracked, advanceSteps()]);

    if (!responseData) {
      setAppState("error");
      setErrorMessage("No response from the analysis server. Make sure the Flask server is running (npm run dev starts it automatically).");
      resetSteps();
      return;
    }

    const data = responseData as AnalysisResponse;

    if (data.status === "error" && data.error) {
      setAppState("error");
      setErrorMessage(data.error);
      resetSteps();
      return;
    }

    const resolveStatus = (s: string | undefined): PipelineStep["status"] => {
      if (!s) return "pending";
      if (s === "OK") return "done";
      if (s.startsWith("ERROR")) return "error";
      if (s.startsWith("SKIPPED")) return "skipped";
      return "done";
    };

    setSteps((prev) =>
      prev.map((s) => ({
        ...s,
        status: resolveStatus(data.steps[s.key]?.status as string),
      }))
    );

    setResult(data);
    setAppState("done");
  };

  const canAnalyze = selectedFile !== null && appState !== "analyzing";
  const showResults = appState === "done" && result !== null;
  const showError = appState === "error";

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <Header />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">

        {/* Page title */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Arteriovenous Ratio Analysis
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Upload a color fundus photograph to compute the AVR microvascular biomarker.
          </p>
        </div>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* ── Left column ─────────────────────────────── */}
          <div className="space-y-5">

            {/* Upload card */}
            <div className="card p-6">
              <div className="card-header">
                <div className="w-7 h-7 rounded-lg bg-sky-50 border border-sky-200 flex items-center justify-center">
                  <FlaskConical size={14} className="text-sky-600" />
                </div>
                <h2 className="text-sm font-semibold text-slate-800">Upload Fundus Image</h2>
              </div>

              <UploadZone
                onFileSelect={handleFileSelect}
                isAnalyzing={appState === "analyzing"}
                selectedFile={selectedFile}
                onClear={handleClear}
              />

              <button
                onClick={handleAnalyze}
                disabled={!canAnalyze}
                suppressHydrationWarning
                className={`
                  mt-4 w-full flex items-center justify-center gap-2 py-3 px-6
                  rounded-xl text-sm font-semibold border transition-all duration-150
                  ${canAnalyze
                    ? "bg-sky-700 hover:bg-sky-800 text-white border-sky-700 shadow-sm active:scale-[0.99]"
                    : "bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed"
                  }
                `}
              >
                {appState === "analyzing" ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Analysing image…
                  </>
                ) : (
                  "Run AVR Analysis"
                )}
              </button>
            </div>

            {/* Pipeline steps */}
            <div className="card p-6">
              <ProgressSteps steps={steps} />
            </div>

            {/* Reference ranges */}
            <div className="card p-6">
              <p className="label-text mb-4">AVR Reference Ranges</p>
              <div className="space-y-3">
                {[
                  {
                    range: "< 0.6",
                    label: "Low AVR",
                    desc: "Arteriolar narrowing — hypertension, cardiovascular risk, sepsis",
                    dot: "bg-red-500",
                    text: "text-red-700",
                  },
                  {
                    range: "0.6 – 0.8",
                    label: "Normal",
                    desc: "Healthy microvascular calibre ratio",
                    dot: "bg-emerald-500",
                    text: "text-emerald-700",
                  },
                  {
                    range: "> 0.8",
                    label: "High AVR",
                    desc: "Venular dilation — inflammation, metabolic syndrome",
                    dot: "bg-amber-500",
                    text: "text-amber-700",
                  },
                ].map((item) => (
                  <div key={item.range} className="flex items-start gap-3">
                    <div className={`w-2 h-2 rounded-full ${item.dot} flex-shrink-0 mt-1.5`} />
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-semibold text-slate-700">{item.range}</span>
                        <span className={`text-xs font-semibold ${item.text}`}>{item.label}</span>
                      </div>
                      <p className="text-xs text-slate-500 mt-0.5">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Right column ────────────────────────────── */}
          <div>
            {showError && (
              <div className="card p-6 border-red-200">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-red-50 border border-red-200 flex items-center justify-center flex-shrink-0">
                    <AlertCircle size={16} className="text-red-600" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-red-800 mb-1">Analysis Failed</h3>
                    <p className="text-sm text-red-600 leading-relaxed">{errorMessage}</p>
                    <button
                      onClick={handleClear}
                      className="mt-4 text-xs text-red-600 hover:text-red-800 underline font-medium"
                    >
                      Clear and try again
                    </button>
                  </div>
                </div>
              </div>
            )}

            {!showResults && !showError && (
              <div className="card h-full min-h-[400px] flex flex-col items-center justify-center text-center p-10">
                <div className="w-16 h-16 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center mb-4">
                  <svg className="w-8 h-8 text-slate-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="9" />
                    <path d="M12 7v5l3 3" strokeLinecap="round" />
                  </svg>
                </div>
                <h3 className="text-base font-semibold text-slate-600 mb-2">
                  Awaiting Analysis
                </h3>
                <p className="text-sm text-slate-400 max-w-xs leading-relaxed">
                  Upload a fundus image and click{" "}
                  <strong className="text-slate-500">Run AVR Analysis</strong> to view results here.
                </p>
              </div>
            )}

            {showResults && result && (
              <div className="animate-fade-in">
                <ResultsPanel data={result} />
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-400">
          <span>SepsisScope · Retinal AVR Pipeline · Knudtson et al. (2003)</span>
          <span className="flex items-center gap-1">
            <AlertCircle size={11} />
            Research use only — not a medical diagnostic device
          </span>
        </div>
      </footer>
    </div>
  );
}
