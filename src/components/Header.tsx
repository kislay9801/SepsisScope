"use client";

import { Eye, BookOpen } from "lucide-react";

export function Header() {
  return (
    <header className="bg-white border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">

          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-sky-700 flex items-center justify-center">
              <Eye className="w-4 h-4 text-white" />
            </div>
            <div>
              <span className="text-slate-900 font-bold text-base tracking-tight leading-none">
                SepsisScope
              </span>
              <p className="text-slate-400 text-[10px] leading-none mt-0.5 hidden sm:block">
                Retinal AVR Analysis
              </p>
            </div>
          </div>

          {/* Right nav */}
          <div className="flex items-center gap-4">
            <a
              href="https://pubmed.ncbi.nlm.nih.gov/12917150/"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500 hover:text-sky-700 transition-colors font-medium"
            >
              <BookOpen size={13} />
              Knudtson 2003
            </a>

            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-50 border border-emerald-200">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span className="text-emerald-700 text-xs font-semibold">Research Tool</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
