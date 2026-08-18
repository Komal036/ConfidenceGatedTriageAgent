"use client";

import { Tag, Search, Wrench, Scale, LucideIcon } from "lucide-react";

export type StageState = "idle" | "active" | "done" | "skipped";

export type StageStatus = {
  classify: StageState;
  retrieve: StageState;
  resolve: StageState;
  judge: StageState;
};

const STAGES: {
  key: keyof StageStatus;
  label: string;
  hint: string;
  icon: LucideIcon;
}[] = [
  { key: "classify", label: "Classify", hint: "category · priority", icon: Tag },
  { key: "retrieve", label: "Retrieve", hint: "KB similarity", icon: Search },
  { key: "resolve", label: "Resolve", hint: "draft · tool call", icon: Wrench },
  { key: "judge", label: "Judge", hint: "escalation gate", icon: Scale },
];

function nodeClasses(state: StageState) {
  if (state === "done")
    return "border-signal-resolved bg-signal-resolved/10 text-signal-resolved";
  if (state === "active")
    return "border-signal-caution bg-signal-caution/10 text-signal-caution";
  if (state === "skipped")
    return "border-ink-line bg-ink-surface text-signal-muted";
  return "border-ink-line bg-ink-surface text-signal-muted";
}

export default function PipelineStages({ status }: { status: StageStatus }) {
  return (
    <div className="w-full">
      <div className="flex items-start">
        {STAGES.map((stage, i) => {
          const state = status[stage.key];
          const isLast = i === STAGES.length - 1;
          const Icon = stage.icon;
          return (
            <div key={stage.key} className="flex items-start flex-1">
              <div className="flex flex-col items-center gap-2.5 shrink-0 w-16">
                <div
                  className={`relative h-10 w-10 rounded-full border flex items-center justify-center transition-all duration-300 ${nodeClasses(
                    state
                  )}`}
                >
                  <Icon size={16} strokeWidth={2} />
                  {state === "active" && (
                    <span className="absolute inset-0 rounded-full border border-signal-caution animate-ping opacity-40" />
                  )}
                </div>
                <div className="text-center">
                  <div
                    className={`text-[11px] font-display tracking-wide transition-colors ${
                      state === "done" || state === "active"
                        ? "text-signal-text"
                        : "text-signal-muted"
                    }`}
                  >
                    {stage.label}
                  </div>
                  <div className="text-[9px] font-mono text-signal-muted mt-0.5 leading-tight">
                    {stage.hint}
                  </div>
                </div>
              </div>
              {!isLast && (
                <div className="flex-1 h-10 flex items-center px-1">
                  <div className="w-full h-px bg-ink-line relative overflow-hidden rounded-full">
                    <div
                      className="h-full bg-signal-resolved transition-all duration-500 ease-out"
                      style={{ width: state === "done" ? "100%" : "0%" }}
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
