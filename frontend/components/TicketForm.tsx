"use client";

import { useState } from "react";
import { Send } from "lucide-react";

export type TicketInput = {
  subject: string;
  description: string;
  product: string;
  channel: string;
};

const CHANNELS = ["chat", "email", "phone", "social media"];

export default function TicketForm({
  onSubmit,
  submitting,
}: {
  onSubmit: (ticket: TicketInput) => void;
  submitting: boolean;
}) {
  const [subject, setSubject] = useState("");
  const [description, setDescription] = useState("");
  const [product, setProduct] = useState("");
  const [channel, setChannel] = useState(CHANNELS[0]);

  const canSubmit = subject.trim() && description.trim() && !submitting;

  return (
    <div className="rounded-lg border border-ink-line bg-ink-surface/40 overflow-hidden">
      <div className="flex items-center justify-between border-b border-ink-line px-5 py-3">
        <span className="text-[11px] uppercase tracking-[0.25em] text-signal-muted font-mono">
          New Ticket
        </span>
        <span className="h-1.5 w-1.5 rounded-full bg-signal-resolved animate-pulse" />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!canSubmit) return;
          onSubmit({ subject, description, product, channel });
        }}
        className="p-5 space-y-5"
      >
        <div>
          <label className="block text-[11px] uppercase tracking-[0.2em] text-signal-muted mb-2 font-mono">
            Subject
          </label>
          <input
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="Cannot connect to WiFi"
            className="w-full rounded-md bg-ink border border-ink-line px-3.5 py-2.5 text-sm text-signal-text placeholder:text-signal-muted/50 font-body focus:border-signal-resolved/60 outline-none transition-colors"
          />
        </div>

        <div>
          <label className="block text-[11px] uppercase tracking-[0.2em] text-signal-muted mb-2 font-mono">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="My laptop keeps disconnecting every few minutes."
            rows={5}
            className="w-full rounded-md bg-ink border border-ink-line px-3.5 py-2.5 text-sm text-signal-text placeholder:text-signal-muted/50 font-body focus:border-signal-resolved/60 outline-none transition-colors resize-none"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-[11px] uppercase tracking-[0.2em] text-signal-muted mb-2 font-mono">
              Product
            </label>
            <input
              value={product}
              onChange={(e) => setProduct(e.target.value)}
              placeholder="Dell XPS"
              className="w-full rounded-md bg-ink border border-ink-line px-3.5 py-2.5 text-sm text-signal-text placeholder:text-signal-muted/50 font-body focus:border-signal-resolved/60 outline-none transition-colors"
            />
          </div>
          <div>
            <label className="block text-[11px] uppercase tracking-[0.2em] text-signal-muted mb-2 font-mono">
              Channel
            </label>
            <select
              value={channel}
              onChange={(e) => setChannel(e.target.value)}
              className="w-full rounded-md bg-ink border border-ink-line px-3.5 py-2.5 text-sm text-signal-text font-body focus:border-signal-resolved/60 outline-none transition-colors"
            >
              {CHANNELS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={!canSubmit}
          className="w-full flex items-center justify-center gap-2 rounded-md bg-signal-resolved text-ink text-sm font-display font-medium tracking-wide py-3 disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 transition-all"
        >
          <Send size={14} strokeWidth={2.5} />
          {submitting ? "Submitting…" : "Submit Ticket"}
        </button>
      </form>
    </div>
  );
}
