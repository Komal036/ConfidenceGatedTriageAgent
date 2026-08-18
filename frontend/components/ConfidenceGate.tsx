"use client";

import { Lock, LockOpen } from "lucide-react";

type ConfidenceGateProps = {
  similarity: number | null;
  threshold: number;
  open: boolean;
  armed: boolean; // false = no result yet, gate sits at rest
};

export default function ConfidenceGate({
  similarity,
  threshold,
  open,
  armed,
}: ConfidenceGateProps) {
  const clamped = similarity !== null ? Math.max(0, Math.min(1, similarity)) : 0;
  const markerPct = clamped * 100;
  const thresholdPct = threshold * 100;
  const leafOffset = armed && open ? 44 : 0;

  return (
    <div className="w-full">
      <div className="flex items-baseline justify-between mb-3">
        <span className="text-[11px] uppercase tracking-[0.25em] text-signal-muted font-mono">
          Confidence Gate
        </span>
        <span className="font-mono text-[11px] text-signal-muted">
          threshold&nbsp;=&nbsp;{threshold.toFixed(2)}
        </span>
      </div>

      {/* Similarity gauge */}
      <div className="relative h-2 rounded-full bg-black/40 border border-ink-line overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 transition-all duration-700 ease-out"
          style={{
            width: `${markerPct}%`,
            background: armed && open
              ? "linear-gradient(90deg, rgba(79,209,165,0.1), #4FD1A5)"
              : "linear-gradient(90deg, rgba(242,166,90,0.1), #F2A65A)",
          }}
        />
        <div
          className="absolute inset-y-0 w-px bg-signal-text/70"
          style={{ left: `${thresholdPct}%` }}
        />
      </div>
      <div className="mt-1.5 flex justify-between font-mono text-[10px] text-signal-muted">
        <span>0.00</span>
        <span className="text-signal-text">
          {similarity !== null ? similarity.toFixed(3) : "—"}
        </span>
        <span>1.00</span>
      </div>

      {/* Blast-door gate */}
      <div className="relative mt-5 h-24 rounded-lg border border-ink-line bg-black/40 overflow-hidden">
        {/* Hazard seam, revealed as leaves part */}
        <div
          className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-10"
          style={{
            background:
              "repeating-linear-gradient(45deg, #F2A65A 0px, #F2A65A 6px, #14181F 6px, #14181F 12px)",
            opacity: armed ? (open ? 0.9 : 0.35) : 0.15,
          }}
        />
        {/* Left leaf */}
        <div
          className="absolute inset-y-0 left-0 w-1/2 border-r border-ink-line transition-transform duration-700 ease-out flex items-center justify-end pr-4"
          style={{
            background: "#1C222C",
            transform: `translateX(-${leafOffset}%)`,
          }}
        >
          <div className="h-8 w-1 rounded-full bg-ink-line" />
        </div>
        {/* Right leaf */}
        <div
          className="absolute inset-y-0 right-0 w-1/2 border-l border-ink-line transition-transform duration-700 ease-out flex items-center justify-start pl-4"
          style={{
            background: "#1C222C",
            transform: `translateX(${leafOffset}%)`,
          }}
        >
          <div className="h-8 w-1 rounded-full bg-ink-line" />
        </div>

        {/* Status readout, centered, above the leaves */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div
            className={`flex items-center gap-2 rounded-full border px-4 py-1.5 backdrop-blur-sm transition-colors duration-500 ${
              !armed
                ? "border-ink-line text-signal-muted bg-ink/60"
                : open
                ? "border-signal-resolved/50 text-signal-resolved bg-ink/70"
                : "border-signal-caution/50 text-signal-caution bg-ink/70"
            }`}
          >
            {!armed ? (
              <>
                <Lock size={14} strokeWidth={2} />
                <span className="font-mono text-[11px] tracking-wide">
                  STANDBY
                </span>
              </>
            ) : open ? (
              <>
                <LockOpen size={14} strokeWidth={2} />
                <span className="font-mono text-[11px] tracking-wide">
                  OPEN — AUTO-RESOLVED
                </span>
              </>
            ) : (
              <>
                <Lock size={14} strokeWidth={2} />
                <span className="font-mono text-[11px] tracking-wide">
                  HELD — ESCALATED
                </span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
