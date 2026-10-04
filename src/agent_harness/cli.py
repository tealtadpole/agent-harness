"""Command line: serve the web app, or check that everything is configured."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-harness", description="LangGraph agent harness")
    parser.add_argument("--config", help="path to config.toml (default: $AGENT_HARNESS_CONFIG, "
                                         "./config.toml, ~/.config/agent-harness/config.toml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the web UI and API")
    sub.add_parser("check", help="check the database, providers and MCP servers")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return _serve(cfg) if args.command == "serve" else asyncio.run(_check(cfg))


def _serve(cfg) -> int:
    import uvicorn

    from .api import FRONTEND_DIST, create_app

    if not FRONTEND_DIST.is_dir():
        logging.warning("No built web UI at %s; run `npm run build` in frontend/ (API only for now)",
                        FRONTEND_DIST)
    if cfg.server.host not in ("127.0.0.1", "localhost", "::1"):
        logging.warning("Listening on %s: there is no login, so anyone who can reach this "
                        "address can use your API keys and tools.", cfg.server.host)
    print(f"Agent harness on http://{cfg.server.host}:{cfg.server.port}", file=sys.stderr)
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="info")
    return 0


async def _check(cfg) -> int:
    from .api import create_app

    ok = True
    app = create_app(cfg)
    try:
        async with app.router.lifespan_context(app):
            state = app.state.harness
            print("Database: connected")
            for server in state.mcp.status():
                mark = "ok " if server["ok"] else "ERR"
                detail = ", ".join(server["tools"]) if server["ok"] else server["error"]
                print(f"MCP {mark} {server['name']}: {detail}")
                ok &= server["ok"]
            for provider in state.providers.values():
                s = await provider.status()
                mark = "ok " if s.available else "-- "
                models = f" models: {', '.join(s.models)}" if s.models else ""
                print(f"{mark} {s.label}: {s.detail or 'ready'}{models}")
            if cfg.jira.enabled:
                from .jira_client import jira_status
                ok_j, detail_j = await jira_status(cfg.jira)
                print(f"{'ok ' if ok_j else '-- '} JIRA: {detail_j}")
                ok &= ok_j
            if cfg.github.enabled:
                from .github_client import github_status
                ok_g, detail_g = await github_status(cfg.github)
                print(f"{'ok ' if ok_g else '-- '} GitHub: {detail_g}")
                ok &= ok_g
            if cfg.workflow.enabled:
                repos = await state.workflow_db.list_platform_repos()
                workflows = await state.workflow_db.list_workflows()
                print(f"ok  Workflows: {len(repos)} platform repo mapping(s), "
                     f"{len(workflows)} workflow(s) on record")
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
