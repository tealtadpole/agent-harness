import { useCallback, useEffect, useMemo, useState } from "react";
import { api, sendMessage, type McpStatus, type Message, type ProviderStatus, type Session, type SessionDetail } from "./api";
import Composer from "./components/Composer";
import MessageList from "./components/MessageList";
import Sidebar from "./components/Sidebar";

interface Live {
  messages: Message[]; // user message + tool calls of the turn in progress
  text: string; // assistant text streamed so far
}

const EMPTY_LIVE: Live = { messages: [], text: "" };

function sessionFromHash(): string | null {
  const m = window.location.hash.match(/^#\/s\/([0-9a-f-]{36})$/);
  return m ? m[1] : null;
}

function loadPick(): { provider: string; model: string } {
  try {
    return JSON.parse(localStorage.getItem("harness.pick") ?? "") ?? { provider: "", model: "" };
  } catch {
    return { provider: "", model: "" };
  }
}

export default function App() {
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [mcp, setMcp] = useState<McpStatus[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(sessionFromHash());
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [live, setLive] = useState<Live>(EMPTY_LIVE);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState(loadPick);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refreshSessions = useCallback(() => api.sessions().then(setSessions).catch((e) => setError(String(e.message ?? e))), []);

  useEffect(() => {
    api
      .providers()
      .then(({ providers, mcp_servers }) => {
        setProviders(providers);
        setMcp(mcp_servers);
        setPick((p) => {
          const current = providers.find((x) => x.name === p.provider && x.available);
          const chosen = current ?? providers.find((x) => x.available) ?? providers[0];
          if (!chosen) return p;
          const model = chosen.models.includes(p.model) ? p.model : chosen.default_model;
          return { provider: chosen.name, model };
        });
      })
      .catch((e) => setError(`Cannot reach the server: ${e.message ?? e}`));
    refreshSessions();
    const onHash = () => setActiveId(sessionFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [refreshSessions]);

  useEffect(() => {
    try {
      localStorage.setItem("harness.pick", JSON.stringify(pick));
    } catch {
      /* storage unavailable */
    }
  }, [pick]);

  const loadDetail = useCallback(async (id: string) => {
    try {
      setDetail(await api.session(id));
    } catch (e) {
      setDetail(null);
      setError((e as Error).message);
      if (window.location.hash) window.location.hash = "";
    }
  }, []);

  useEffect(() => {
    setLive(EMPTY_LIVE);
    if (activeId) loadDetail(activeId);
    else setDetail(null);
  }, [activeId, loadDetail]);

  // A reply that was still running when the page loaded (e.g. after a refresh): poll until done.
  useEffect(() => {
    if (!detail?.running || streaming) return;
    const t = setTimeout(() => loadDetail(detail.id), 2000);
    return () => clearTimeout(t);
  }, [detail, streaming, loadDetail]);

  const select = (id: string | null) => {
    window.location.hash = id ? `/s/${id}` : "";
    setActiveId(id);
    setSidebarOpen(false);
  };

  const newChat = () => select(null);

  const send = async (text: string) => {
    setError(null);
    let session: Session | SessionDetail | null = detail;
    if (!session) {
      try {
        session = await api.createSession(pick.provider, pick.model);
      } catch (e) {
        setError((e as Error).message);
        return;
      }
      setSessions((s) => [session as Session, ...s]);
      window.location.hash = `/s/${session.id}`;
      setActiveId(session.id);
      setDetail({ ...session, messages: [], running: true });
    }
    const id = session.id;
    setStreaming(true);
    setLive({ messages: [{ id: "pending-user", role: "user", content: text, pending: true }], text: "" });
    try {
      await sendMessage(id, text, (e) => {
        switch (e.type) {
          case "user":
            setLive((l) => ({ ...l, messages: [e.message, ...l.messages.filter((m) => m.id !== "pending-user")] }));
            break;
          case "title":
            setSessions((s) => s.map((x) => (x.id === id ? { ...x, title: e.title } : x)));
            setDetail((d) => (d && d.id === id ? { ...d, title: e.title } : d));
            break;
          case "token":
            setLive((l) => ({ ...l, text: l.text + e.text }));
            break;
          case "tool_start":
            setLive((l) => ({ ...l, messages: [...l.messages, { ...e.message, pending: true }] }));
            break;
          case "tool_end":
            setLive((l) => ({
              ...l,
              messages: l.messages.map((m) =>
                m.tool_call_id === e.tool_call_id ? { ...m, content: e.output, pending: false } : m,
              ),
            }));
            break;
          case "notice":
          case "error":
            setLive((l) => ({
              ...l,
              messages: [...l.messages, { id: `err-${Date.now()}`, role: "error", content: e.type === "error" ? e.message : e.text }],
            }));
            break;
        }
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStreaming(false);
      await loadDetail(id);
      setLive(EMPTY_LIVE);
      refreshSessions();
    }
  };

  const rename = async (id: string, title: string) => {
    try {
      const s = await api.renameSession(id, title);
      setSessions((list) => list.map((x) => (x.id === id ? { ...x, title: s.title } : x)));
      setDetail((d) => (d && d.id === id ? { ...d, title: s.title } : d));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteSession(id);
      setSessions((list) => list.filter((x) => x.id !== id));
      if (id === activeId) select(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const messages = useMemo(() => {
    const saved = detail?.messages ?? [];
    const current = [...live.messages];
    if (streaming) current.push({ id: "live-assistant", role: "assistant", content: live.text, pending: true });
    return [...saved, ...current];
  }, [detail, live, streaming]);

  const provider = providers.find((p) => p.name === (detail?.provider ?? pick.provider));
  const unavailable = providers.filter((p) => !p.available);
  const brokenMcp = mcp.filter((m) => !m.ok);

  return (
    <div className={`app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <Sidebar
        providers={providers}
        mcp={mcp}
        sessions={sessions}
        activeId={activeId}
        pick={pick}
        onPick={setPick}
        onSelect={select}
        onNew={newChat}
        onRename={rename}
        onDelete={remove}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="main">
        <header className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open chats">
            ☰
          </button>
          <div className="topbar-title">
            <h1>{detail?.title ?? "New chat"}</h1>
            <span className="badge">
              {provider?.label ?? detail?.provider ?? pick.provider} · {detail?.model ?? pick.model}
            </span>
          </div>
        </header>

        {(error || (!detail && (unavailable.length > 0 || brokenMcp.length > 0))) && (
          <div className="banners">
            {error && (
              <div className="banner banner-error">
                {error}
                <button className="link-btn" onClick={() => setError(null)}>
                  Dismiss
                </button>
              </div>
            )}
            {!detail &&
              unavailable.map((p) => (
                <div key={p.name} className="banner">
                  <strong>{p.label}:</strong> {p.detail || "not available"}
                </div>
              ))}
            {!detail &&
              brokenMcp.map((m) => (
                <div key={m.name} className="banner banner-error">
                  <strong>MCP server “{m.name}” failed:</strong> {m.error}
                </div>
              ))}
          </div>
        )}

        <MessageList messages={messages} streaming={streaming || !!detail?.running} empty={!detail} mcp={mcp} />
        <Composer
          disabled={streaming || !!detail?.running || (!detail && !provider?.available)}
          placeholder={
            detail?.running && !streaming
              ? "A reply is still being generated…"
              : !detail && !provider?.available
                ? "Pick an available model in the sidebar to start"
                : "Ask anything… (Enter to send, Shift+Enter for a new line)"
          }
          onSend={send}
        />
      </main>
    </div>
  );
}
