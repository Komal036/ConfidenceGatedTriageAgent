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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Runtime shape check for /submit-ticket's response.
 *
 * Why this exists: TicketResult above is a compile-time-only contract.
 * `res.json()` returns `Promise<any>`, so casting that straight to
 * `Promise<TicketResult>` (as submitTicket used to) gives TypeScript no
 * way to catch a mismatch between what the type declares and what the
 * backend actually sends -- that gap is exactly how `escalated` and
 * `escalation_reason` went missing silently for a while: the type said
 * they existed, the backend didn't send them, nothing complained, and
 * `!result.escalated` (undefined) quietly evaluated to `true` for every
 * ticket. See README's Key Learnings section for the full story.
 *
 * This function is deliberately narrow: it checks that each field is
 * present and has the right primitive type, not that the values are
 * semantically sensible (e.g. it doesn't check category is one of
 * VALID_CATEGORIES -- that's the backend's job, and duplicating it here
 * would just be two places to keep in sync). The goal is purely to turn
 * "a field silently doesn't exist" into "a thrown error naming exactly
 * which field is missing," right at the call site, instead of that gap
 * surfacing three components later as a mysteriously wrong UI state.
 */
function assertIsTicketResult(data: unknown): asserts data is TicketResult {
  if (typeof data !== "object" || data === null) {
    throw new Error(
      `submitTicket: expected an object in the response body, got ${typeof data}`,
    );
  }

  const d = data as Record<string, unknown>;

  const requiredStrings: (keyof TicketResult)[] = [
    "id",
    "subject",
    "description",
    "category",
    "priority",
    "status",
    "escalation_reason",
  ];
  for (const field of requiredStrings) {
    if (typeof d[field] !== "string") {
      throw new Error(
        `submitTicket: expected "${field}" to be a string, got ${typeof d[field]}. ` +
          `This usually means the backend's TicketResponse schema and the frontend's ` +
          `TicketResult type have drifted apart -- check app/schemas.py.`,
      );
    }
  }

  if (typeof d.escalated !== "boolean") {
    throw new Error(
      `submitTicket: expected "escalated" to be a boolean, got ${typeof d.escalated}. ` +
        `If this is undefined, the backend likely stopped sending this field -- ` +
        `check TicketResponse in app/schemas.py and submit_ticket() in app/main.py.`,
    );
  }

  if (d.status !== "resolved" && d.status !== "no_match") {
    throw new Error(
      `submitTicket: expected "status" to be "resolved" or "no_match", got "${String(d.status)}"`,
    );
  }

  const nullableStrings: (keyof TicketResult)[] = [
    "matched_issue",
    "draft_resolution",
    "tool_called",
  ];
  for (const field of nullableStrings) {
    const value = d[field];
    if (value !== null && typeof value !== "string") {
      throw new Error(
        `submitTicket: expected "${field}" to be a string or null, got ${typeof value}`,
      );
    }
  }

  if (d.match_similarity !== null && typeof d.match_similarity !== "number") {
    throw new Error(
      `submitTicket: expected "match_similarity" to be a number or null, got ${typeof d.match_similarity}`,
    );
  }
}

export async function submitTicket(ticket: TicketInput): Promise<TicketResult> {
  const res = await fetch(`${API_BASE_URL}/submit-ticket`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(ticket),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Request failed (${res.status}): ${text || res.statusText}`,
    );
  }

  const data: unknown = await res.json();
  assertIsTicketResult(data);
  return data;
}
