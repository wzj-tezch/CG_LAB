"""Resume large file download with HTTP Range support."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def resume(url: str, dest: Path, chunk_mb: int = 4, retries: int = 20) -> bool:
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # consolidate existing partial data
    start = 0
    if part.exists():
        start = max(start, part.stat().st_size)
    if dest.exists():
        start = max(start, dest.stat().st_size)
        if start > 0 and (not part.exists() or part.stat().st_size < start):
            with open(dest, "rb") as src, open(part, "wb") as dst:
                while True:
                    buf = src.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)

    total_all = 0
    done = start
    for attempt in range(1, retries + 1):
        headers = {"User-Agent": "Mozilla/5.0"}
        if done > 0:
            headers["Range"] = f"bytes={done}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                if "Content-Range" in resp.headers:
                    total_all = int(resp.headers["Content-Range"].split("/")[-1])
                elif resp.headers.get("Content-Length"):
                    total_all = done + int(resp.headers["Content-Length"])
                print(f"attempt {attempt}: resume {done}/{total_all or '?'} bytes")
                t0 = time.time()
                mode = "ab" if done else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(chunk_mb * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total_all and done % (64 * 1024 * 1024) < chunk_mb * 1024 * 1024:
                            rate = (done - start) / max(time.time() - t0, 1e-3) / 1e6
                            print(f"  {done}/{total_all} ({100 * done / total_all:.2f}%) {rate:.2f} MB/s", flush=True)
            if total_all and done >= total_all - 4096:
                os.replace(part, dest)
                print("COMPLETE", dest, done)
                return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
            print(f"attempt {attempt} failed at {done}: {exc}")
            time.sleep(min(5 * attempt, 30))
            continue
    print("INCOMPLETE", done, "expected", total_all)
    return False


if __name__ == "__main__":
    ok = resume(
        "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz",
        MODELS / "hmr2_data.tar.gz",
    )
    raise SystemExit(0 if ok else 1)
