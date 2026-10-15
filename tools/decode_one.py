"""NEZAVISNI dekoder: dobija SAMO zapis (i fiksni model = ovaj kod).

Upotreba: python3 tools/decode_one.py zapis.bin izlaz.bin [--junk N]
--junk N: zapisu se pre dekodovanja doda N nasumičnih bajtova; dekoder mora
pročitati tačno len(zapis) bajtova (dokaz da ne koristi dužinu fajla).
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from igc import codec  # noqa: E402
from igc.film import film_bytes  # noqa: E402


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
    rec_path, out_path = sys.argv[1:3]
    junk = int(sys.argv[sys.argv.index("--junk") + 1]) if "--junk" in sys.argv else 0
    rec = Path(rec_path).read_bytes()
    res = {"model_id": codec.model_id(), "rec_len": len(rec)}
    t = time.perf_counter()
    try:
        meta, payload, used = codec.decode(rec + os.urandom(junk))
        Path(out_path).write_bytes(film_bytes(meta, payload))
        res.update(ok=True, consumed=used)
    except Exception as e:  # neuspeh dekodovanja je validan rezultat eksperimenta
        res.update(ok=False, error=f"{type(e).__name__}: {e}")
    res["time_s"] = time.perf_counter() - t
    res["maxrss_mb"] = peak_rss_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
