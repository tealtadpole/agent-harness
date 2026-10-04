import { useState } from "react";
import type { McpStatus, ProviderStatus, Session } from "../api";

interface Props {
  providers: ProviderStatus[];
  mcp: McpStatus[];
  sessions: Session[];
  activeId: string | null;
  pick: { provider: string; model: string };
  onPick: (p: { provider: string; model: string }) => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

function timeAgo(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d ago`;
  return new Date(iso).toLocaleDateString();
}

export default function Sidebar(props: Props) {
  const { providers, mcp, sessions, activeId, pick } = props;
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const provider = providers.find((p) => p.name === pick.provider);

  const commit = (id: string) => {
    const title = draft.trim();
    if (title) props.onRename(id, title);
    setEditing(null);
  };

  return (
    <>
      <div className="scrim" onClick={props.onClose} />
      <aside className="sidebar">
        <div className="sidebar-head">
          <span className="brand">Agent Harness</span>
          <button className="btn-primary" onClick={props.onNew}>
            + New chat
          </button>
        </div>

        <div className="picker">
          <label>
            <span>Provider</span>
            <select
              value={pick.provider}
              onChange={(e) => {
                const p = providers.find((x) => x.name === e.target.value);
                props.onPick({ provider: e.target.value, model: p?.default_model ?? "" });
              }}
            >
              {providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.label}
                  {p.available ? "" : " (unavailable)"}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Model</span>
            <select
              value={pick.model}
              disabled={!provider?.models.length}
              onChange={(e) => props.onPick({ ...pick, model: e.target.value })}
            >
              {(provider?.models.length ? provider.models : [pick.model || "default"]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">Used for new chats. Each chat keeps the model it started with.</p>
        </div>

        <nav className="session-list" aria-label="Chats">
          {sessions.length === 0 && <p className="hint pad">No chats yet.</p>}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session ${s.id === activeId ? "active" : ""}`}
              onClick={() => editing !== s.id && props.onSelect(s.id)}
            >
              {editing === s.id ? (
                <input
                  autoFocus
                  className="rename"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={() => commit(s.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commit(s.id);
                    if (e.key === "Escape") setEditing(null);
                  }}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <>
                  <div className="session-title" title={s.title}>
                    {s.title}
                  </div>
                  <div className="session-meta">
                    {s.provider} · {s.model} · {timeAgo(s.updated_at)}
                  </div>
                  <div className="session-actions">
                    <button
                      className="icon-btn"
                      aria-label="Rename chat"
                      title="Rename"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDraft(s.title);
                        setEditing(s.id);
                      }}
                    >
                      ✎
                    </button>
                    <button
                      className="icon-btn"
                      aria-label="Delete chat"
                      title="Delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Delete “${s.title}”? This removes its history.`)) props.onDelete(s.id);
                      }}
                    >
                      ×
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </nav>

        <footer className="sidebar-foot">
          <span className="hint">Tools</span>
          {mcp.length === 0 && <span className="hint">No MCP servers configured</span>}
          {mcp.map((m) => (
            <div key={m.name} className="mcp" title={m.ok ? m.tools.join(", ") : (m.error ?? "")}>
              <span className={`dot ${m.ok ? "ok" : "bad"}`} />
              {m.name} {m.ok ? `(${m.tools.length} tools)` : "(failed)"}
            </div>
          ))}
        </footer>
      </aside>
    </>
  );
}
