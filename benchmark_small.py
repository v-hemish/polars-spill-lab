import json
import resource
import sys
import threading
import time
from pathlib import Path

import polars as pl

label = sys.argv[1]
stop = threading.Event()
peak = [0]

def monitor():
    while not stop.is_set():
        total = 0
        for file in Path("/var/tmp").rglob("*.ipc"):
            try:
                total += file.stat().st_size
            except FileNotFoundError:
                pass
        peak[0] = max(peak[0], total)
        stop.wait(0.05)

thread = threading.Thread(target=monitor, daemon=True)
thread.start()
print(f"{label}: Polars {pl.__version__}", flush=True)
started = time.perf_counter()
completed = False

try:
    (
        pl.scan_parquet("data/part-*.parquet")
        .head(5_000_000)
        .sort(["key", "id"])
        .sink_parquet(f"results/{label}.parquet", engine="streaming")
    )
    completed = True
finally:
    stop.set()
    thread.join()
    stats = {
        "label": label,
        "polars": pl.__version__,
        "completed": completed,
        "seconds": round(time.perf_counter() - started, 2),
        "peak_process_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "sampled_peak_spill_mib": round(peak[0] / 1024**2, 1),
    }
    Path(f"results/{label}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
