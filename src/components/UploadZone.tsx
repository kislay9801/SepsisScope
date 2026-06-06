"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, X, FileImage, ImagePlus } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  isAnalyzing: boolean;
  selectedFile: File | null;
  onClear: () => void;
}

const SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"];
const MAX_SIZE_MB = 20;

export function UploadZone({ onFileSelect, isAnalyzing, selectedFile, onClear }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validate = useCallback(
    (file: File) => {
      setError(null);
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!SUPPORTED_FORMATS.includes(ext)) {
        setError(`Unsupported format: ${ext}. Accepted: ${SUPPORTED_FORMATS.join(", ")}`);
        return;
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        setError(`File too large (max ${MAX_SIZE_MB} MB).`);
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) validate(file);
    },
    [validate]
  );

  const handleInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validate(file);
    },
    [validate]
  );

  const handleClear = () => {
    setPreview(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    onClear();
  };

  return (
    <div className="space-y-3">
      <div
        onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !selectedFile && !isAnalyzing && fileInputRef.current?.click()}
        className={`
          relative rounded-xl border-2 border-dashed overflow-hidden transition-all duration-150
          ${isAnalyzing ? "cursor-not-allowed" : "cursor-pointer"}
          ${isDragging
            ? "border-sky-400 bg-sky-50"
            : selectedFile
              ? "border-emerald-300 bg-emerald-50/40"
              : "border-slate-300 hover:border-sky-400 hover:bg-sky-50/40 bg-white"
          }
        `}
      >
        {/* Preview */}
        {preview && (
          <div className="relative">
            <img
              src={preview}
              alt="Fundus preview"
              className="w-full h-48 object-contain bg-slate-900/5"
            />
            {isAnalyzing && (
              <div className="absolute inset-0 bg-white/60 flex flex-col items-center justify-center">
                <div className="w-6 h-6 border-2 border-slate-300 border-t-sky-600 rounded-full animate-spin mb-2" />
                <span className="text-xs font-medium text-slate-600">Analysing…</span>
              </div>
            )}
            {!isAnalyzing && (
              <button
                onClick={(e) => { e.stopPropagation(); handleClear(); }}
                className="absolute top-2 right-2 p-1 rounded-md bg-white border border-slate-200 shadow-sm text-slate-500 hover:text-red-500 hover:border-red-200 transition-colors"
              >
                <X size={13} />
              </button>
            )}
          </div>
        )}

        {/* Empty state */}
        {!preview && (
          <div className="flex flex-col items-center justify-center py-10 px-6 text-center">
            <div className={`w-12 h-12 rounded-xl border flex items-center justify-center mb-3 transition-colors ${isDragging ? "bg-sky-100 border-sky-200" : "bg-slate-100 border-slate-200"}`}>
              {isDragging
                ? <ImagePlus size={20} className="text-sky-600" />
                : <Upload size={20} className="text-slate-400" />
              }
            </div>
            <p className="text-sm font-medium text-slate-700 mb-1">
              {isDragging ? "Release to upload" : "Drop image here or click to browse"}
            </p>
            <p className="text-xs text-slate-400">
              PNG · JPG · TIFF · BMP · PPM — up to {MAX_SIZE_MB} MB
            </p>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-xs text-red-600">
          <X size={12} />
          {error}
        </div>
      )}

      {/* File info */}
      {selectedFile && !error && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-50 border border-slate-200">
          <FileImage size={14} className="text-slate-400 flex-shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="text-xs text-slate-700 truncate font-medium">{selectedFile.name}</p>
            <p className="text-xs text-slate-400">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept={SUPPORTED_FORMATS.join(",")}
        onChange={handleInput}
      />
    </div>
  );
}
