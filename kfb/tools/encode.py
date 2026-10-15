"""Enkoder u zasebnom procesu: python3 -m kfb.tools.encode ENKODER film rec BUDZET_S"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from igc.film import parse_film  # noqa: E402
from kfb import encoders  # noqa: E402


def vmhwm_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return -1.0


def main() -> None:
    name, film, rec, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    enc = encoders.load_all()[name]
    meta, payload = parse_film(Path(film).read_bytes())
    t = time.perf_counter()
    r = enc.fn(meta, payload, budget)
    dt = time.perf_counter() - t
    res = {"ok": r is not None, "time_s": dt, "rss_mb": vmhwm_mb()}
    if r is not None:
        Path(rec).write_bytes(r)
        res["size"] = len(r)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
