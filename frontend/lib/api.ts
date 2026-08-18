export type TicketResult = {
  id: string;
  subject: string;
  description: string;
  category: string;
  priority: string;
  status: "resolved" | "no_match";
  matched_issue: string | null;
  match_similarity: number | null;
  draft_resolution: string | null;
  tool_called: string | null;
  escalated: boolean;
  escalation_reason: string;
};

export type TicketInput = {
  subject: string;
  description: string;
  product: string;
  channel: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function submitTicket(ticket: TicketInput): Promise<TicketResult> {
  const res = await fetch(`${API_BASE_URL}/submit-ticket`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(ticket),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Request failed (${res.status}): ${text || res.statusText}`
    );
  }

  return res.json();
}
