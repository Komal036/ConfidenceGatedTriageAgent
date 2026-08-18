"use client";

import { useState } from "react";
import { Activity } from "lucide-react";
import TicketForm, { TicketInput } from "@/components/TicketForm";
import PipelineStages, { StageStatus } from "@/components/PipelineStages";
import ConfidenceGate from "@/components/ConfidenceGate";
import ResultPanel from "@/components/ResultPanel";
import TiltCard from "@/components/TiltCard";
import { submitTicket, TicketResult } from "@/lib/api";

const THRESHOLD = 0.6;

const IDLE_STAGES: StageStatus = {
  classify: "idle",
  retrieve: "idle",
  resolve: "idle",
  judge: "idle",
};

export default function Home() {
  const [stages, setStages] = useState<StageStatus>(IDLE_STAGES);
  const [result, setResult] = useState<TicketResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [ticketCount, setTicketCount] = useState(0);
  const [parallax, setParallax] = useState({ x: 0, y: 0 });

  function handlePageMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const x = (e.clientX / window.innerWidth - 0.5) * 12; // px range, subtle
    const y = (e.clientY / window.innerHeight - 0.5) * 12;
    setParallax({ x, y });
  }

  async function handleSubmit(ticket: TicketInput) {
    setSubmitting(true);
    setError(null);
    setResult(null);
    setStages({ classify: "active", retrieve: "idle", resolve: "idle", judge: "idle" });

    const advance = (next: Partial<StageStatus>) =>
      setStages((prev) => ({ ...prev, ...next }));

    const timers = [
      setTimeout(() => advance({ classify: "done", retrieve: "active" }), 500),
      setTimeout(() => advance({ retrieve: "done", resolve: "active" }), 1100),
      setTimeout(() => advance({ resolve: "done", judge: "active" }), 1700),
    ];

    try {
      const res = await submitTicket(ticket);
      timers.forEach(clearTimeout);
      setStages({
        classify: "done",
        retrieve: "done",
        resolve: res.status === "resolved" ? "done" : "skipped",
        judge: "done",
      });
      setResult(res);
      setTicketCount((n) => n + 1);
    } catch (err) {
      timers.forEach(clearTimeout);
      setStages(IDLE_STAGES);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  const armed = result !== null;
  const gateOpen = armed ? !result!.escalated : false;

  return (
    <main className="min-h-screen relative overflow-hidden" onMouseMove={handlePageMouseMove}>
      {/* Faint technical grid background, with subtle parallax */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.04] transition-transform duration-200 ease-out"
        style={{
          backgroundImage:
            "linear-gradient(#E8ECEF 1px, transparent 1px), linear-gradient(90deg, #E8ECEF 1px, transparent 1px)",
          backgroundSize: "48px 48px",
          maskImage:
            "radial-gradient(ellipse 80% 60% at 50% 0%, black 40%, transparent 90%)",
          transform: `translate(${parallax.x}px, ${parallax.y}px)`,
        }}
      />

      {/* Header bar */}
      <header className="relative border-b border-ink-line">
        <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="h-2 w-2 rounded-full bg-signal-resolved" />
            <span className="font-mono text-xs tracking-[0.2em] text-signal-muted">
              CONFIDENCE-GATED TRIAGE AGENT
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-signal-muted">
            <Activity size={13} strokeWidth={2} />
            <span className="font-mono text-xs">
              {ticketCount} ticket{ticketCount === 1 ? "" : "s"} this session
            </span>
          </div>
        </div>
      </header>

      <div className="relative mx-auto max-w-6xl px-6 py-12 md:py-16">
        {/* Hero */}
        <div className="mb-12 max-w-2xl">
          <h1 className="font-display text-3xl md:text-[2.75rem] font-medium tracking-tight text-signal-text leading-[1.1]">
            A system that knows when
            <br />
            <span className="text-signal-resolved">not</span> to act.
          </h1>
          <p className="mt-4 text-signal-muted font-body max-w-xl leading-relaxed text-[15px]">
            Four agents classify, search a knowledge base, draft a
            resolution, and decide — with a visible, auditable threshold —
            whether to act autonomously or hand off to a human.
          </p>
        </div>

        {/* Two-pane console */}
        <div className="grid lg:grid-cols-[380px_1fr] gap-6 items-start">
          {/* Left: input panel, sticky on desktop */}
          <div className="lg:sticky lg:top-8">
            <TiltCard className="rounded-lg shadow-[0_20px_50px_-15px_rgba(0,0,0,0.6)]">
              <TicketForm onSubmit={handleSubmit} submitting={submitting} />
            </TiltCard>
          </div>

          {/* Right: live pipeline, gate, results */}
          <div className="space-y-6">
            <div className="rounded-lg border border-ink-line bg-ink-surface/40 p-6">
              <PipelineStages status={stages} />
            </div>

            <TiltCard className="rounded-lg shadow-[0_20px_50px_-15px_rgba(0,0,0,0.6)]">
              <div className="rounded-lg border border-ink-line bg-ink-surface/40 p-6">
                <ConfidenceGate
                  similarity={result?.match_similarity ?? null}
                  threshold={THRESHOLD}
                  open={gateOpen}
                  armed={armed}
                />
              </div>
            </TiltCard>

            {error && (
              <div className="rounded-md border border-signal-caution/40 bg-signal-caution/5 p-4">
                <p className="font-mono text-sm text-signal-caution">{error}</p>
                <p className="font-body text-xs text-signal-muted mt-2">
                  Make sure the FastAPI backend is running and
                  NEXT_PUBLIC_API_URL points to it.
                </p>
              </div>
            )}

            {result && (
              <div className="rounded-lg border border-ink-line bg-ink-surface/40 p-6">
                <ResultPanel result={result} />
              </div>
            )}

            {!result && !error && (
              <div className="rounded-lg border border-dashed border-ink-line p-8 text-center">
                <p className="font-mono text-xs text-signal-muted">
                  Submit a ticket to see the pipeline in action.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
