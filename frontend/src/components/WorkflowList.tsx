import { useState } from "react";
import type { Workflow } from "../api";

function timeAgo(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

const STATUS_LABEL: Record<string, string> = {
  pending: "Starting…", running: "Running", awaiting_approval: "Awaiting approval",
  merging: "Merging", advancing: "Advancing", completed: "Completed", failed: "Failed",
};

export default function WorkflowList({
  workflows,
  onStart,
  onSelect,
  error,
}: {
  workflows: Workflow[];
  onStart: (ticketKey: string) => Promise<void>;
  onSelect: (id: string) => void;
  error: string | null;
}) {
  const [ticketKey, setTicketKey] = useState("");
  const [starting, setStarting] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const key = ticketKey.trim();
    if (!key || starting) return;
    setStarting(true);
    try {
      await onStart(key);
      setTicketKey("");
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="messages">
      <div className="thread" style={{ gap: 20 }}>
        <form className="composer" style={{ padding: 0, margin: 0, maxWidth: "none" }} onSubmit={submit}>
          <input
            className="rename"
            style={{ flex: 1 }}
            placeholder="JIRA ticket key, e.g. PROJ-123"
            value={ticketKey}
            onChange={(e) => setTicketKey(e.target.value)}
            disabled={starting}
          />
          <button className="btn-primary" type="submit" disabled={starting || !ticketKey.trim()}>
            {starting ? "Starting…" : "Start workflow"}
          </button>
        </form>

        {error && <div className="banner banner-error">{error}</div>}

        {workflows.length === 0 && <p className="hint">No workflows yet.</p>}
        <div style={{ display: "grid", gap: 8 }}>
          {workflows.map((w) => (
            <div key={w.id} className="session" style={{ cursor: "pointer" }} onClick={() => onSelect(w.id)}>
              <div className="session-title">
                {w.ticket_key}: {w.ticket_summary || "(no summary)"}
              </div>
              <div className="session-meta">
                {STATUS_LABEL[w.status] ?? w.status} · stage {w.current_stage} · {w.platform} · {timeAgo(w.updated_at)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
