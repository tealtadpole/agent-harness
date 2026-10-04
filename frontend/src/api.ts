export type Role = "user" | "assistant" | "tool" | "error";

export interface Message {
  id: number | string;
  role: Role;
  content: string;
  tool_name?: string | null;
  tool_call_id?: string | null;
  tool_input?: unknown;
  created_at?: string;
  pending?: boolean; // UI-only: tool still running / text still streaming
}

export interface Session {
  id: string;
  title: string;
  provider: string;
  model: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface SessionDetail extends Session {
  messages: Message[];
  running: boolean;
}

export interface ProviderStatus {
  name: string;
  label: string;
  available: boolean;
  detail: string;
  models: string[];
  default_model: string;
}

export interface McpStatus {
  name: string;
  ok: boolean;
  tools: string[];
  error: string | null;
}

export type StreamEvent =
  | { type: "user"; message: Message }
  | { type: "title"; title: string }
  | { type: "token"; text: string }
  | { type: "tool_start"; message: Message }
  | { type: "tool_end"; tool_call_id: string; output: string }
  | { type: "assistant"; message: Message }
  | { type: "notice"; text: string }
  | { type: "error"; message: string }
  | { type: "done" };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export interface PlatformRepo {
  platform: string;
  repo_owner: string;
  repo_name: string;
  base_branch: string;
  clone_url: string;
  created_at: string;
  updated_at: string;
}

export type WorkflowStatus =
  | "pending" | "running" | "awaiting_approval" | "merging" | "advancing" | "completed" | "failed";
export type Stage = "spec" | "plan" | "tasks" | "implement";
export type StageStatus = "pending" | "running" | "pr_open" | "approved" | "merged" | "failed";

export interface WorkflowStage {
  id: number;
  workflow_id: string;
  stage: Stage;
  branch: string;
  pr_number: number | null;
  pr_url: string | null;
  status: StageStatus;
  artifact_path: string | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
}

export interface Workflow {
  id: string;
  ticket_key: string;
  ticket_summary: string;
  platform: string;
  repo_owner: string;
  repo_name: string;
  base_branch: string;
  clone_url: string;
  slug: string;
  status: WorkflowStatus;
  current_stage: Stage | "done";
  error: string | null;
  created_at: string;
  updated_at: string;
  stages: WorkflowStage[];
}

export interface WorkflowEvent {
  id: number;
  workflow_id: string;
  stage: Stage | null;
  type:
    | "stage_start" | "agent_text" | "tool_call" | "stage_pr_opened" | "stage_approved"
    | "stage_merged" | "workflow_completed" | "workflow_failed" | "error";
  payload: Record<string, unknown>;
  created_at: string;
}

export const api = {
  providers: () => request<{ providers: ProviderStatus[]; mcp_servers: McpStatus[] }>("/api/providers"),
  sessions: () => request<Session[]>("/api/sessions"),
  session: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),
  createSession: (provider: string, model: string) =>
    request<Session>("/api/sessions", { method: "POST", body: JSON.stringify({ provider, model }) }),
  renameSession: (id: string, title: string) =>
    request<Session>(`/api/sessions/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteSession: (id: string) => request<void>(`/api/sessions/${id}`, { method: "DELETE" }),

  platformRepos: () => request<PlatformRepo[]>("/api/platform-repos"),
  upsertPlatformRepo: (repo: Omit<PlatformRepo, "created_at" | "updated_at">) =>
    request<PlatformRepo>("/api/platform-repos", { method: "POST", body: JSON.stringify(repo) }),
  deletePlatformRepo: (platform: string) =>
    request<void>(`/api/platform-repos/${encodeURIComponent(platform)}`, { method: "DELETE" }),

  workflows: () => request<Workflow[]>("/api/workflows"),
  workflow: (id: string) => request<Workflow>(`/api/workflows/${id}`),
  startWorkflow: (ticketKey: string) =>
    request<Workflow>("/api/workflows", { method: "POST", body: JSON.stringify({ ticket_key: ticketKey }) }),
};

/** Stream a workflow's event log (past + live) using the browser's native EventSource, which
 * auto-reconnects with Last-Event-ID -- the backend sends real `id:` lines for exactly this. */
export function streamWorkflow(id: string, onEvent: (e: WorkflowEvent) => void): () => void {
  const es = new EventSource(`/api/workflows/${id}/events`);
  es.onmessage = (m) => onEvent(JSON.parse(m.data) as WorkflowEvent);
  return () => es.close();
}

/** POST a message and call `onEvent` for each Server-Sent Event until the turn is done. */
export async function sendMessage(
  sessionId: string,
  content: string,
  onEvent: (e: StreamEvent) => void,
): Promise<void> {
  const res = await fetch(`/api/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      }
    }
  }
}
