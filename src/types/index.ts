export type StepStatus = "pending" | "running" | "done" | "error" | "skipped";

export interface StepResult {
  status: string;
  [key: string]: unknown;
}

export interface DiscInfo {
  cx: number | null;
  cy: number | null;
  r: number | null;
  confidence: number | null;
  method: string | null;
  flag: string;
}

export interface SegmentCounts {
  total: number;
  in_zone: number;
  rejected: number;
  arterioles: number;
  venules: number;
  uncertain: number;
}

export interface VesselAudit {
  detected: number;
  rejected_outside_zone: number;
  in_zone: number;
  rejected_uncertain: number;
  classified_arterioles: number;
  classified_venules: number;
  rejected_width_outlier: number;
  used_arterioles: number;
  used_venules: number;
  used_total: number;
}

export interface Reliability {
  level: "high" | "moderate" | "low" | "n/a";
  score: number;
  reasons: string[];
}

export interface AnalysisResult {
  AVR: number | null;
  CRAE: number | null;
  CRVE: number | null;
  flag: "normal" | "low_avr" | "high_avr" | "insufficient_arterioles" | "insufficient_venules" | "insufficient_segments" | "zero_crve" | "unknown";
  interpretation: string;
  segments: SegmentCounts;
  disc: DiscInfo;
  vessel_audit?: VesselAudit;
  reliability?: Reliability;
}

export interface AnalysisImages {
  original?: string;
  overlay?: string;
  disc?: string;
  zone?: string;
  classified?: string;
  audit?: string;
}

export interface AnalysisResponse {
  status: "success" | "error";
  steps: {
    step1?: StepResult;
    step2?: StepResult;
    step3?: StepResult;
    step4?: StepResult;
    step5?: StepResult;
  };
  result: AnalysisResult;
  images: AnalysisImages;
  error?: string;
}

export interface PipelineStep {
  id: number;
  key: "step1" | "step2" | "step3" | "step4" | "step5";
  label: string;
  description: string;
  icon: string;
  status: StepStatus;
}
