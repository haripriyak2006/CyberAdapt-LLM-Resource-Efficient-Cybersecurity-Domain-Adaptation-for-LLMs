#!/usr/bin/env python3
"""
start.py — CyberAdapt-LLM one-command startup launcher.

Usage:
    python start.py              # launch backend + frontend
    python start.py --backend    # backend only
    python start.py --frontend   # frontend only
    python start.py --check      # health check only
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).parent
_py_exe = "python.exe" if sys.platform == "win32" else "python"
_venv_bin = "Scripts" if sys.platform == "win32" else "bin"
VENV_PYTHON = ROOT / ".venv" / _venv_bin / _py_exe
NODE = "npm.cmd" if sys.platform == "win32" else "npm"

BACKEND_PORT = 8000
FRONTEND_PORT = 3000

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner():
    print(f"""
{CYAN}{BOLD}
  ██████╗██╗   ██╗██████╗ ███████╗██████╗  █████╗ ██████╗  █████╗ ██████╗ ████████╗
 ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝
 ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝███████║██║  ██║███████║██████╔╝   ██║
 ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗██╔══██║██║  ██║██╔══██║██╔═══╝    ██║
 ╚██████╗   ██║   ██████╔╝███████╗██║  ██║██║  ██║██████╔╝██║  ██║██║        ██║
  ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝        ╚═╝
{RESET}
{BOLD}  CyberAdapt-LLM{RESET} — Resource-Efficient Cybersecurity Domain Adaptation
  Phase 11 · Hackathon Demo
""")


def info(msg: str):
    print(f"  {CYAN}►{RESET} {msg}")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}⚠{RESET} {msg}")


def err(msg: str):
    print(f"  {RED}✗{RESET} {msg}")


def wait_for_url(url: str, label: str, timeout: int = 60) -> bool:
    """Poll until URL responds 200 or timeout."""
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    ok(f"{label} is up → {url}")
                    return True
        except Exception:
            pass
        attempt += 1
        print(f"    waiting... ({attempt}s)", end="\r")
        time.sleep(1)
    err(f"{label} did not respond within {timeout}s")
    return False


def check_python():
    if VENV_PYTHON.exists():
        return VENV_PYTHON
    err("Virtual environment not found at .venv/")
    err("Run this first: python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt")
    sys.exit(1)


def check_node_modules():
    nm = ROOT / "frontend" / "node_modules"
    if not nm.exists():
        info("Installing frontend dependencies...")
        subprocess.run([NODE, "install"], cwd=ROOT / "frontend", check=True)


def health_check_only():
    backend_url = f"http://localhost:{BACKEND_PORT}/health"
    frontend_url = f"http://localhost:{FRONTEND_PORT}"
    print()
    info("Checking backend health...")
    try:
        with urllib.request.urlopen(backend_url, timeout=3) as r:
            import json
            data = json.loads(r.read())
            ok(f"Backend: {data.get('status', 'ok')} | {data.get('app', '')} v{data.get('version', '')}")
    except Exception as e:
        err(f"Backend not responding: {e}")

    info("Checking frontend...")
    try:
        with urllib.request.urlopen(frontend_url, timeout=3) as r:
            ok(f"Frontend: HTTP {r.status}")
    except Exception as e:
        err(f"Frontend not responding: {e}")
    print()


def start_backend(python: Path) -> subprocess.Popen:
    info("Starting FastAPI backend on port 8000...")
    env = os.environ.copy()
    env["APP_ENV"] = env.get("APP_ENV", "development")
    proc = subprocess.Popen(
        [str(python), "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", str(BACKEND_PORT), "--reload"],
        cwd=ROOT,
        env=env,
    )
    return proc


def start_frontend() -> subprocess.Popen:
    info("Starting Next.js frontend on port 3000...")
    proc = subprocess.Popen(
        [NODE, "run", "dev"],
        cwd=ROOT / "frontend",
    )
    return proc


def main():
    banner()
    parser = argparse.ArgumentParser(description="CyberAdapt-LLM launcher")
    parser.add_argument("--backend",  action="store_true", help="Backend only")
    parser.add_argument("--frontend", action="store_true", help="Frontend only")
    parser.add_argument("--check",    action="store_true", help="Health check only")
    args = parser.parse_args()

    if args.check:
        health_check_only()
        return

    procs = []
    python = check_python()

    try:
        if not args.frontend:
            procs.append(start_backend(python))
            ok("Backend process started (PID %d)" % procs[-1].pid)
            wait_for_url(f"http://localhost:{BACKEND_PORT}/health", "Backend", timeout=60)

        if not args.backend:
            check_node_modules()
            procs.append(start_frontend())
            ok("Frontend process started (PID %d)" % procs[-1].pid)
            wait_for_url(f"http://localhost:{FRONTEND_PORT}", "Frontend", timeout=90)

        print(f"""
{GREEN}{BOLD}  ✓ CyberAdapt-LLM is running!{RESET}

    Dashboard  →  {BOLD}http://localhost:{FRONTEND_PORT}{RESET}
    API Docs   →  {BOLD}http://localhost:{BACKEND_PORT}/docs{RESET}
    Health     →  {BOLD}http://localhost:{BACKEND_PORT}/health{RESET}

  {YELLOW}Press Ctrl+C to stop all services.{RESET}
""")

        # Wait for any child process to exit
        while True:
            for p in procs:
                if p.poll() is not None:
                    err(f"Process {p.pid} exited unexpectedly (code {p.returncode})")
                    raise KeyboardInterrupt
            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Shutting down...{RESET}")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print(f"  {GREEN}All services stopped.{RESET}\n")


if __name__ == "__main__":
    main()
