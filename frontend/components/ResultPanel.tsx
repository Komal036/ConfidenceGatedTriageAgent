"use client";

import { BookOpen, FileText, Wrench, Gavel, LucideIcon } from "lucide-react";
import type { TicketResult } from "@/lib/api";

function Badge({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "default" | "resolved" | "caution";
}) {
  const toneClass =
    tone === "resolved"
      ? "border-signal-resolved/40 text-signal-resolved"
      : tone === "caution"
      ? "border-signal-caution/40 text-signal-caution"
      : "border-ink-line text-signal-muted";
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-mono ${toneClass}`}
    >
      {children}
    </span>
  );
}

function Section({
  icon: Icon,
  label,
  children,
}: {
  icon: LucideIcon;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <Icon size={12} strokeWidth={2} />
        <span className="text-[11px] uppercase tracking-[0.2em] text-signal-muted font-mono">
          {label}
        </span>
      </div>
      {children}
    </div>
  );
}

export default function ResultPanel({ result }: { result: TicketResult }) {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        <Badge>{result.category}</Badge>
        <Badge tone={result.priority === "Critical" ? "caution" : "default"}>
          priority · {result.priority}
        </Badge>
        <Badge tone={result.escalated ? "caution" : "resolved"}>
          {result.escalated ? "escalated" : "auto-resolved"}
        </Badge>
      </div>

      {result.matched_issue && (
        <Section icon={BookOpen} label="Matched Knowledge Base Entry">
          <p className="font-body text-sm text-signal-text leading-relaxed">
            {result.matched_issue}
          </p>
        </Section>
      )}

      {result.draft_resolution && (
        <Section icon={FileText} label="Draft Resolution">
          <p className="font-body text-sm text-signal-text leading-relaxed">
            {result.draft_resolution}
          </p>
        </Section>
      )}

      {result.tool_called && (
        <Section icon={Wrench} label="Tool Invoked">
          <code className="font-mono text-sm text-signal-resolved">
            {result.tool_called}()
          </code>
        </Section>
      )}

      <Section icon={Gavel} label="Escalation Judge Reasoning">
        <p className="font-mono text-xs leading-relaxed text-signal-muted bg-black/30 border border-ink-line rounded-md p-3.5">
          {result.escalation_reason}
        </p>
      </Section>
    </div>
  );
}
