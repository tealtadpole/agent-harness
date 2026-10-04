import { useEffect, useRef } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { McpStatus, Message } from "../api";

function ToolCall({ m }: { m: Message }) {
  const input = m.tool_input && typeof m.tool_input === "object" ? JSON.stringify(m.tool_input, null, 2) : String(m.tool_input ?? "");
  const summary =
    m.tool_input && typeof m.tool_input === "object"
      ? Object.values(m.tool_input as Record<string, unknown>)
          .map((v) => (typeof v === "string" ? v : JSON.stringify(v)))
          .join(", ")
      : "";
  return (
    <details className="tool">
      <summary>
        <span className={`dot ${m.pending ? "busy" : "ok"}`} />
        <code>{m.tool_name}</code>
        <span className="tool-summary">{summary}</span>
        {m.pending && <span className="hint">running…</span>}
      </summary>
      <div className="tool-body">
        <div className="hint">Input</div>
        <pre>{input}</pre>
        <div className="hint">Output</div>
        <pre>{m.pending ? "…" : m.content || "(empty)"}</pre>
      </div>
    </details>
  );
}

function Bubble({ m }: { m: Message }) {
  if (m.role === "tool") return <ToolCall m={m} />;
  if (m.role === "error") return <div className="msg msg-error">{m.content}</div>;
  if (m.role === "user") return <div className="msg msg-user">{m.content}</div>;
  return (
    <div className="msg msg-assistant">
      {m.content ? (
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{ a: (props) => <a {...props} target="_blank" rel="noopener noreferrer" /> }}
        >
          {m.content}
        </Markdown>
      ) : (
        <span className="typing" aria-label="Thinking">
          <i />
          <i />
          <i />
        </span>
      )}
    </div>
  );
}

export default function MessageList({
  messages,
  streaming,
  empty,
  mcp,
}: {
  messages: Message[];
  streaming: boolean;
  empty: boolean;
  mcp: McpStatus[];
}) {
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const last = messages[messages.length - 1];

  // Follow new output, unless the user has scrolled up to read.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    if (nearBottom || !streaming) end.current?.scrollIntoView({ block: "end" });
  }, [messages.length, last?.content, streaming]);

  if (empty && messages.length === 0) {
    const tools = mcp.filter((m) => m.ok).flatMap((m) => m.tools);
    return (
      <div className="messages empty" ref={box}>
        <div className="welcome">
          <h2>Start a conversation</h2>
          <p>Pick a provider and model in the sidebar, then ask a question.</p>
          {tools.length > 0 && (
            <p className="hint">
              Available tools: {tools.map((t) => <code key={t}>{t}</code>)}
            </p>
          )}
        </div>
        <div ref={end} />
      </div>
    );
  }

  return (
    <div className="messages" ref={box}>
      <div className="thread">
        {messages.map((m) => (
          <Bubble key={m.id} m={m} />
        ))}
        <div ref={end} />
      </div>
    </div>
  );
}
