"""Replay a historical FD001 training trajectory through the running backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.replay import load_training_trajectory, replay_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-id", type=int, default=81, help="FD001 training unit to replay")
    parser.add_argument("--asset-id", help="Backend asset ID (default: fd001-unit-N)")
    parser.add_argument("--asset-name", help="Asset display name")
    parser.add_argument("--cycles-per-second", type=float, default=1.0, help="Replay rate; 0 is unthrottled")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--token", default=os.getenv("RELIABILITY_API_TOKEN"), help="Bearer token (or RELIABILITY_API_TOKEN)")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "source" / "CMAPSSData")
    parser.add_argument("--no-resume", action="store_true", help="Submit the full trajectory instead of skipping stored cycles")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    asset_id = args.asset_id or f"fd001-unit-{args.unit_id}"
    asset_name = args.asset_name or f"FD001 Replay Unit {args.unit_id}"
    trajectory = load_training_trajectory(args.raw_dir, args.unit_id)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=60.0) as client:
        result = await replay_trajectory(
            client,
            trajectory,
            asset_id=asset_id,
            asset_name=asset_name,
            unit_id=args.unit_id,
            cycles_per_second=args.cycles_per_second,
            resume=not args.no_resume,
            bearer_token=args.token,
        )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
