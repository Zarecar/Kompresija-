"""Enkoder u ZASEBNOM procesu: čita original sa diska, upisuje zapis.

Upotreba: python3 tools/encode_one.py METODA film.bin zapis.bin [--leaky-dir DIR]
Ispisuje JSON: veličina zapisa, vreme, vršna memorija (RSS celog procesa).
"""
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from igc import codec  # noqa: E402
from igc.film import parse_film  # noqa: E402


def peak_rss_mb() -> float:
    """Vršni RSS OVOG procesa (VmHWM). ru_maxrss na Linuxu nasleđuje RSS roditelja
    preko fork+exec, pa bi merio memoriju eksperimenta, a ne dekodera."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    method, film_path, rec_path = sys.argv[1:4]
    kw = {}
    if "--leaky-dir" in sys.argv:
        kw["leaky_dir"] = sys.argv[sys.argv.index("--leaky-dir") + 1]
    meta, payload = parse_film(Path(film_path).read_bytes())
    t = time.perf_counter()
    rec = codec.encode(method, meta, payload, **kw)
    dt = time.perf_counter() - t
    res = {"ok": rec is not None, "time_s": dt,
           "maxrss_mb": peak_rss_mb()}
    if rec is not None:
        Path(rec_path).write_bytes(rec)
        res["size"] = len(rec)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
