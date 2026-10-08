import argparse
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

p = argparse.ArgumentParser()
p.add_argument("action", choices=["generate", "sort", "verify"])
p.add_argument("--rows", type=int, default=30_000_000)
p.add_argument("--label", default="run")
p.add_argument("--engine", default="streaming")
a = p.parse_args()

if a.action == "generate":
    rng = np.random.default_rng(42)
    for start in range(0, a.rows, 250_000):
        end = min(start + 250_000, a.rows)
        frame = pl.DataFrame({
            "id": np.arange(start, end, dtype=np.int64),
            "key": rng.integers(0, 2**62, end - start, dtype=np.int64),
            **{f"value_{i}": rng.random(end - start) for i in range(7)},
        })
        frame.write_parquet(f"data/part-{start:09d}.parquet")
        print(f"Generated {end:,}/{a.rows:,}", flush=True)

elif a.action == "sort":
    temp = Path("spill") / a.label
    temp.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    peak = [0]

    def monitor():
        while not stop.is_set():
            total = 0
            for file in temp.rglob("*"):
                try:
                    if file.is_file():
                        total += file.stat().st_size
                except FileNotFoundError:
                    pass
            peak[0] = max(peak[0], total)
            stop.wait(0.05)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    print(f"Polars {pl.__version__}; engine={a.engine}", flush=True)
    started = time.perf_counter()
    try:
        (
            pl.scan_parquet("data/part-*.parquet")
            .sort(["key", "id"])
            .sink_parquet(
                f"results/{a.label}.parquet",
                engine=a.engine,
            )
        )
    finally:
        stop.set()
        thread.join()

    result = {
        "label": a.label,
        "polars": pl.__version__,
        "engine": a.engine,
        "seconds": round(time.perf_counter() - started, 2),
        "peak_process_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "sampled_peak_temp_mib": round(peak[0] / 1024**2, 1),
    }
    Path(f"results/{a.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

else:
    for file in sorted(Path("results").glob("*.parquet")):
        check = pl.scan_parquet(file).select(
            pl.len().alias("rows"),
            pl.col("id").sum().alias("id_sum"),
            (
                (pl.col("key") < pl.col("key").shift(1))
                | (
                    (pl.col("key") == pl.col("key").shift(1))
                    & (pl.col("id") < pl.col("id").shift(1))
                )
            ).fill_null(False).sum().alias("order_errors"),
        ).collect(engine="streaming").row(0, named=True)

        assert check["rows"] == a.rows, check
        assert check["id_sum"] == a.rows * (a.rows - 1) // 2, check
        assert check["order_errors"] == 0, check
        print(file.name, check, "PASS", flush=True)
