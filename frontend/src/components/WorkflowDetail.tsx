import { useEffect, useState } from "react";
import { streamWorkflow, type Message, type Stage, type Workflow, type WorkflowEvent } from "../api";
import MessageList from "./MessageList";

const STAGES: Stage[] = ["spec", "plan", "tasks", "implement"];

const STAGE_LABEL: Record<Stage, string> = {
  spec: "Spec", plan: "Plan", tasks: "Tasks", implement: "Implement",
};

const STAGE_STATUS_LABEL: Record<string, string> = {
  pending: "Not started", running: "Running…", pr_open: "Waiting for PR approval",
  approved: "Approved, merging…", merged: "Merged", failed: "Failed",
};

function StageCard({ workflow, stage }: { workflow: Workflow; stage: Stage }) {
  const row = workflow.stages.find((s) => s.stage === stage);
  const active = workflow.current_stage === stage && workflow.status !== "completed" && workflow.status !== "failed";
  return (
    <div className={`session ${active ? "active" : ""}`} style={{ cursor: "default" }}>
      <div className="session-title">{STAGE_LABEL[stage]}</div>
      <div className="session-meta">{row ? (STAGE_STATUS_LABEL[row.status] ?? row.status) : "Not started"}</div>
      {row?.pr_url && (
        <a href={row.pr_url} target="_blank" rel="noopener noreferrer" className="hint">
          PR #{row.pr_number} ↗
        </a>
      )}
    </div>
  );
}

function eventsToMessages(events: WorkflowEvent[]): Message[] {
  return events
    .filter((e) => e.type === "agent_text" || e.type === "workflow_failed" || e.type === "error")
    .map((e) => ({
      id: e.id,
      role: e.type === "agent_text" ? "assistant" : "error",
      content: String(e.payload.text ?? (e.type === "workflow_failed" ? "Workflow failed." : e.type)),
    }));
}

export default function WorkflowDetail({ workflow }: { workflow: Workflow }) {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);

  useEffect(() => {
    setEvents([]);
    return streamWorkflow(workflow.id, (e) => setEvents((list) => [...list, e]));
  }, [workflow.id]);

  const running = workflow.status !== "completed" && workflow.status !== "failed";

  return (
    <div className="messages">
      <div className="thread" style={{ gap: 20 }}>
        <div>
          <h2 style={{ margin: "0 0 4px", fontSize: 18 }}>
            {workflow.ticket_key}: {workflow.ticket_summary}
          </h2>
          <p className="hint">
            {workflow.repo_owner}/{workflow.repo_name} · base {workflow.base_branch} · platform {workflow.platform}
          </p>
          {workflow.error && <div className="banner banner-error">{workflow.error}</div>}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
          {STAGES.map((stage) => (
            <StageCard key={stage} workflow={workflow} stage={stage} />
          ))}
        </div>

        <MessageList messages={eventsToMessages(events)} streaming={running} empty={events.length === 0} mcp={[]} />
      </div>
    </div>
  );
}
