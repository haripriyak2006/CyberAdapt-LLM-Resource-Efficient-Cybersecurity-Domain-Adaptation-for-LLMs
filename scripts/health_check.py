"""
scripts/health_check.py
Standalone CLI health-check script for CyberAdapt-LLM.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --host 0.0.0.0 --port 8000
    python scripts/health_check.py --url http://my-server:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def build_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def check_health(url: str, timeout: int = 5) -> dict:
    """
    Perform a GET request to the health endpoint.

    Returns the parsed JSON body.
    Raises SystemExit on failure.
    """
    print(f">> Checking health at: {url}")
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            return data
    except urllib.error.URLError as exc:
        print(f"\nERROR: Could not connect to {url}")
        print(f"   Reason: {exc.reason}")
        print("\n   Is the server running?  Try:")
        print("     uvicorn backend.main:app --reload\n")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"\nERROR: Server responded but body is not valid JSON: {exc}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="CyberAdapt-LLM health check")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--url", default=None, help="Full URL override (ignores --host/--port)")
    parser.add_argument("--timeout", type=int, default=5, help="Request timeout in seconds")
    args = parser.parse_args()

    url = args.url or build_url(args.host, args.port)
    data = check_health(url, timeout=args.timeout)

    # Pretty-print result
    print("\n-- Health Response --------------------------------------")
    print(json.dumps(data, indent=2))
    print("---------------------------------------------------------")

    if data.get("status") == "ok":
        print(f"\nOK - Service is healthy  ({data.get('service')} v{data.get('version')})\n")
        sys.exit(0)
    else:
        print(f"\nFAIL - Service reported unhealthy status: {data.get('status')}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
