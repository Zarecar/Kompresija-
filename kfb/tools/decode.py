"""NEZAVISNI dekoder: python3 -m kfb.tools.decode rec out [--junk N]

Dobija samo zapis; dodaje N nasumičnih bajtova; zapis mora biti samoograničavajući.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from igc.film import film_bytes  # noqa: E402
from kfb import record  # noqa: E402
from kfb.tools.encode import vmhwm_mb  # noqa: E402


def main() -> None:
    rec_path, out_path = sys.argv[1], sys.argv[2]
    junk = int(sys.argv[sys.argv.index("--junk") + 1]) if "--junk" in sys.argv else 0
    rec = Path(rec_path).read_bytes()
    res = {"model_id": record.model_id(), "rec_len": len(rec)}
    t = time.perf_counter()
    try:
        meta, payload, used = record.decode(rec + os.urandom(junk))
        Path(out_path).write_bytes(film_bytes(meta, payload))
        res.update(ok=True, consumed=used)
    except Exception as e:  # neuspeh je validan rezultat
        res.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
    res["time_s"] = time.perf_counter() - t
    res["rss_mb"] = vmhwm_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
