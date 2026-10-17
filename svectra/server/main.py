"""
CLI entry point for running the SvectraDB server.
"""

import argparse
import os
import sys
import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="SvectraDB: High-Performance Vector Database Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host network interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./svectra_data",
        help="Directory path for vector data persistence (default: ./svectra_data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Set data dir env var before starting
    os.environ["SVECTRA_DATA_DIR"] = os.path.abspath(args.data_dir)

    print("=" * 64)
    print("  🚀 SVECTRADB - HIGH-PERFORMANCE VECTOR DATABASE")
    print(f"  • Listening on:  http://{args.host}:{args.port}")
    print(f"  • Web Dashboard: http://localhost:{args.port}/")
    print(f"  • Swagger Docs:  http://localhost:{args.port}/docs")
    print(f"  • Data Storage:  {os.environ['SVECTRA_DATA_DIR']}")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    uvicorn.run(
        "svectra.server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        app_dir=root_dir,
    )


if __name__ == "__main__":
    main()
