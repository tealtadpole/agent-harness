import { useState } from "react";
import type { PlatformRepo } from "../api";

const EMPTY = { platform: "", repo_owner: "", repo_name: "", base_branch: "main", clone_url: "" };

export default function PlatformRepos({
  repos,
  onUpsert,
  onDelete,
  error,
}: {
  repos: PlatformRepo[];
  onUpsert: (repo: typeof EMPTY) => Promise<void>;
  onDelete: (platform: string) => Promise<void>;
  error: string | null;
}) {
  const [draft, setDraft] = useState(EMPTY);
  const [saving, setSaving] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!draft.platform.trim() || !draft.repo_owner.trim() || !draft.repo_name.trim() || !draft.clone_url.trim()) return;
    setSaving(true);
    try {
      await onUpsert(draft);
      setDraft(EMPTY);
    } finally {
      setSaving(false);
    }
  };

  const set = (k: keyof typeof EMPTY) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setDraft((d) => ({ ...d, [k]: e.target.value }));

  return (
    <div className="messages">
      <div className="thread" style={{ gap: 20 }}>
        <div>
          <h2 style={{ margin: "0 0 4px", fontSize: 18 }}>Platform repos</h2>
          <p className="hint">Maps a JIRA ticket's "Platform" field to the repo a workflow should target.</p>
        </div>

        {error && <div className="banner banner-error">{error}</div>}

        <div style={{ display: "grid", gap: 8 }}>
          {repos.length === 0 && <p className="hint">No repos mapped yet.</p>}
          {repos.map((r) => (
            <div key={r.platform} className="session" style={{ cursor: "default" }}>
              <div className="session-title">{r.platform}</div>
              <div className="session-meta">
                {r.repo_owner}/{r.repo_name} · base {r.base_branch} · {r.clone_url}
              </div>
              <div className="session-actions" style={{ display: "flex" }}>
                <button className="icon-btn" aria-label="Delete mapping" title="Delete" onClick={() => onDelete(r.platform)}>
                  ×
                </button>
              </div>
            </div>
          ))}
        </div>

        <form onSubmit={submit} style={{ display: "grid", gap: 8, maxWidth: 480 }}>
          <input className="rename" placeholder="Platform (matches the JIRA field value)" value={draft.platform} onChange={set("platform")} />
          <input className="rename" placeholder="Repo owner (e.g. acme)" value={draft.repo_owner} onChange={set("repo_owner")} />
          <input className="rename" placeholder="Repo name (e.g. webapp)" value={draft.repo_name} onChange={set("repo_name")} />
          <input className="rename" placeholder="Base branch" value={draft.base_branch} onChange={set("base_branch")} />
          <input className="rename" placeholder="Clone URL (https://...)" value={draft.clone_url} onChange={set("clone_url")} />
          <button className="btn-primary" type="submit" disabled={saving}>
            {saving ? "Saving…" : "Add / update mapping"}
          </button>
        </form>
      </div>
    </div>
  );
}
