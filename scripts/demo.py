#!/usr/bin/env python3
"""One-command local demo: FastAPI API + web app.

    python scripts/demo.py              # API on :8000, built web app (vite preview) on :5173
    python scripts/demo.py --dev        # vite dev server instead of a production build
    python scripts/demo.py --offline    # no network access needed (see below)
    python scripts/demo.py --check      # headless smoke of the whole stack, then exit (make demo-check)

Prerequisite: ``make setup`` (Python venv + ``npm ci``). No API keys are used anywhere.

Offline mode: the demo itself never needs the internet (all servers bind to 127.0.0.1; the web
build is local tsc + vite). ``--offline`` additionally forces ``HPE_LLM_PROVIDER=none`` and the script
never runs ``npm install``/``npm ci`` (dependencies must already be installed by ``make setup``).
Stored runs and experiment results are served from ``reports/`` — real outputs produced earlier by
this repository's code (``source="stored_real_output"``), not hand-authored data.

The script refuses to start if a port is already in use, so its checks can never silently talk to
some other server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never via proxy


def _http(method: str, url: str, body: Any | None = None, timeout: float = 120.0) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = r.read()
            ctype = r.headers.get("Content-Type", "")
            return r.status, (json.loads(raw) if "json" in ctype else raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def _wait_for(url: str, timeout: float, proc: subprocess.Popen[bytes] | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with _OPENER.open(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.3)
    return False


def _port_free(port: int) -> bool:
    """True if nothing is bound to 127.0.0.1:port (so a health check can't hit someone else's server)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _require_free(port: int, what: str) -> None:
    if not _port_free(port):
        raise RuntimeError(
            f"port {port} for the {what} is already in use; stop the other process or pass "
            f"--{'api' if what == 'API' else 'web'}-port"
        )


def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _npm() -> str:
    npm = shutil.which("npm")
    if not npm:
        raise SystemExit("npm not found: install Node.js 22 (see README) or run with --api-only")
    return npm


class Stack:
    def __init__(self, args: argparse.Namespace, data_dir: Path) -> None:
        self.args = args
        self.data_dir = data_dir
        self.procs: list[tuple[str, subprocess.Popen[bytes]]] = []
        self.api_url = f"http://127.0.0.1:{args.api_port}"
        self.web_url = f"http://127.0.0.1:{args.web_port}"

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "packages"), str(ROOT / "apps" / "api"), env.get("PYTHONPATH", "")]
        )
        env["HPE_DATA_DIR"] = str(self.data_dir)
        env.setdefault("HPE_REPORTS_DIR", str(ROOT / "reports"))
        env["HPE_CORS_ORIGINS"] = ",".join(
            [f"http://127.0.0.1:{self.args.web_port}", f"http://localhost:{self.args.web_port}"]
        )
        env["VITE_API_BASE_URL"] = self.api_url
        if self.args.offline:
            env["HPE_LLM_PROVIDER"] = "none"
        return env

    def _spawn(self, name: str, cmd: list[str], cwd: Path) -> subprocess.Popen[bytes]:
        log = None if self.args.verbose else subprocess.DEVNULL
        p = subprocess.Popen(cmd, cwd=cwd, env=self._env(), stdout=log, stderr=log, start_new_session=True)
        self.procs.append((name, p))
        return p

    def start_api(self) -> None:
        cmd = [
            _python(),
            "-m",
            "uvicorn",
            "hpe_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.args.api_port),
        ]
        _require_free(self.args.api_port, "API")
        p = self._spawn("api", cmd, ROOT)
        if not _wait_for(self.api_url + "/health", 60, p) or p.poll() is not None:
            raise RuntimeError(f"API did not become healthy on {self.api_url} (exit code {p.poll()})")

    def start_web(self) -> None:
        npm = _npm()
        if not (WEB / "node_modules").exists():
            raise RuntimeError("apps/web/node_modules missing: run `make setup` first")
        if self.args.dev:
            cmd = [
                npm,
                "run",
                "dev",
                "--",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.args.web_port),
                "--strictPort",
            ]
        else:
            dist = WEB / "dist" / "index.html"
            if not self.args.no_build:  # `npm run build` (tsc + vite) is local; it never fetches packages
                subprocess.run(
                    [npm, "run", "build"],
                    cwd=WEB,
                    env=self._env(),
                    check=True,
                    stdout=None if self.args.verbose else subprocess.DEVNULL,
                )
            elif not dist.exists():
                raise RuntimeError("apps/web/dist missing; build once with `cd apps/web && npm run build`")
            cmd = [
                npm,
                "run",
                "preview",
                "--",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.args.web_port),
                "--strictPort",
            ]
        _require_free(self.args.web_port, "web app")
        p = self._spawn("web", cmd, WEB)
        if not _wait_for(self.web_url + "/", 60, p) or p.poll() is not None:
            raise RuntimeError(f"web app did not start on {self.web_url} (exit code {p.poll()})")

    def stop(self) -> None:
        for _name, p in reversed(self.procs):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _name, p in reversed(self.procs):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


SPEC_19 = {
    "user_id": "demo-check-user",
    "consent_given": True,
    "question": "Should I take a higher-paying job, continue in my current role, or attempt a startup?",
    "options": [
        {
            "option_id": "stay",
            "label": "Continue in my current role",
            "kind": "current_path",
            "inputs": {"monthly_income": 5000},
        },
        {
            "option_id": "job",
            "label": "Take the higher-paying job",
            "kind": "new_job",
            "inputs": {"monthly_income": 6200},
        },
        {
            "option_id": "startup",
            "label": "Attempt a startup",
            "kind": "start_project",
            "inputs": {"monthly_income": 0},
        },
    ],
    "values": [
        {"key": "income", "label": "Income", "weight": 0.6},
        {"key": "stability", "label": "Security", "weight": 0.7},
        {"key": "autonomy", "label": "Autonomy", "weight": 0.9},
        {"key": "relationships", "label": "Relationships", "weight": 0.5},
        {"key": "learning", "label": "Growth", "weight": 0.8},
    ],
    "constraints": {"savings": 24000, "monthly_expenses": 3500},
    "horizon_years": 3,
}


def run_checks(stack: Stack, include_web: bool) -> list[tuple[str, bool, str]]:
    api = stack.api_url
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)

    code, body = _http("GET", api + "/health")
    check(
        "GET /health",
        code == 200 and isinstance(body, dict) and body.get("status") == "ok",
        f"{code} {body if isinstance(body, dict) else ''}",
    )
    for path in ("/runs", "/research/claims", "/benchmarks", "/evidence", "/experiments"):
        code, body = _http("GET", api + path)
        check(
            f"GET {path}",
            code == 200 and isinstance(body, list),
            f"{code}, {len(body) if isinstance(body, list) else body!s:.80} items",
        )
    code, body = _http("POST", api + "/scenarios", SPEC_19)
    ok = (
        code == 200
        and isinstance(body, dict)
        and len(body.get("scenarios", [])) == 3
        and all(s.get("label") == "illustrative scenario" for s in body["scenarios"])
    )
    check(
        "POST /scenarios (spec §19, consent)",
        ok,
        f"{code}, {len(body.get('scenarios', [])) if isinstance(body, dict) else body!s:.120} scenarios",
    )
    code, body = _http("POST", api + "/scenarios", {**SPEC_19, "consent_given": False})
    check("POST /scenarios without consent is refused", code == 400, str(code))
    cfg = {"seed": 7, "n_agents": 25, "years": 5, "policy_class": "developmental"}
    code, body = _http("POST", api + "/runs", {"config": cfg, "mode": "interactive"})
    run_id = body.get("run_id") if isinstance(body, dict) else None
    check(
        "POST /runs interactive (25 agents, developmental)", code == 201 and bool(run_id), f"{code} {run_id}"
    )
    if run_id:
        code, body = _http("POST", api + f"/runs/{run_id}/step", {"ticks": 3, "include_frames": True})
        ok = code == 200 and isinstance(body, dict) and body.get("tick", 0) >= 3
        check(
            "POST /runs/{id}/step x3",
            ok,
            f"{code}, tick {body.get('tick') if isinstance(body, dict) else '?'}, population {body.get('population') if isinstance(body, dict) else '?'}",
        )
        code, body = _http("GET", api + f"/runs/{run_id}/agents?limit=1")
        aid = (
            body["items"][0]["agent_id"]
            if code == 200 and isinstance(body, dict) and body.get("items")
            else None
        )
        check(
            "GET /runs/{id}/agents",
            bool(aid),
            f"{code} total={body.get('total') if isinstance(body, dict) else '?'}",
        )
        if aid:
            code, body = _http("GET", api + f"/runs/{run_id}/agents/{aid}")
            check(
                "GET /runs/{id}/agents/{agent_id}",
                code == 200 and isinstance(body, dict) and "skill_curve" in body,
                str(code),
            )
        code, _ = _http("POST", api + f"/runs/{run_id}/control", {"action": "pause"})
        check("POST /runs/{id}/control pause", code == 200, str(code))
    if include_web:
        code, body = _http("GET", stack.web_url + "/")
        html = body.decode(errors="replace") if isinstance(body, bytes) else str(body)
        check("GET web /", code == 200 and '<div id="root"' in html, str(code))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-port", type=int, default=int(os.environ.get("HPE_API_PORT", 8000)))
    ap.add_argument("--web-port", type=int, default=5173)
    ap.add_argument("--dev", action="store_true", help="use the vite dev server")
    ap.add_argument("--no-build", action="store_true", help="serve the existing apps/web/dist")
    ap.add_argument("--offline", action="store_true", help="no network needed; never runs npm build/install")
    ap.add_argument("--api-only", action="store_true", help="start only the API")
    ap.add_argument("--check", action="store_true", help="headless smoke test of the stack, then exit")
    ap.add_argument(
        "--data-dir", default=None, help="runtime data dir (default ./data/demo; temp dir with --check)"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    tmp = tempfile.TemporaryDirectory(prefix="hpe-demo-check-") if args.check else None
    data_dir = Path(args.data_dir) if args.data_dir else (Path(tmp.name) if tmp else ROOT / "data" / "demo")
    data_dir.mkdir(parents=True, exist_ok=True)
    stack = Stack(args, data_dir)
    try:
        print(f"starting API on {stack.api_url} (data: {data_dir})", flush=True)
        stack.start_api()
        if not args.api_only:
            print(
                f"starting web app on {stack.web_url} ({'dev' if args.dev else 'preview build'})", flush=True
            )
            stack.start_web()
        if args.check:
            print("demo-check:", flush=True)
            results = run_checks(stack, include_web=not args.api_only)
            failed = [r for r in results if not r[1]]
            print(f"demo-check: {len(results) - len(failed)}/{len(results)} checks passed", flush=True)
            return 1 if failed else 0
        print(
            "\nHuman Possibility Engine demo is running (synthetic research prototype; not validated for "
            "predicting human futures)\n"
            f"  web app:       {stack.web_url}/\n"
            f"  investor demo: {stack.web_url}/demo\n"
            f"  API docs:      {stack.api_url}/docs\n"
            "Press Ctrl-C to stop.",
            flush=True,
        )
        while all(p.poll() is None for _, p in stack.procs):
            time.sleep(0.5)
        dead = [n for n, p in stack.procs if p.poll() is not None]
        print(f"process exited: {', '.join(dead)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopping…", flush=True)
        return 0
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"demo failed: {e}", file=sys.stderr)
        return 1
    finally:
        stack.stop()
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
