"""Zajednički okvir: enkodovanje i dekodovanje u odvojenim procesima."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = sys.executable
BUDGETS = [1024, 4096, 16384, 65536]


def _run(args: list[str], env: dict | None = None) -> dict:
    e = dict(os.environ)
    e.pop("IGC_LEAKY_DIR", None)
    if env:
        e.update(env)
    p = subprocess.run([PY, *args], capture_output=True, text=True, env=e, timeout=7200)
    if p.returncode != 0:
        return {"ok": False, "error": p.stderr.strip()[-500:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def baseline_rss_mb() -> float:
    code = ("import sys;sys.path.insert(0,%r);import igc.codec;"
            "print([int(l.split()[1])/1024 for l in open('/proc/self/status') if l.startswith('VmHWM:')][0])"
            ) % str(ROOT)
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    return float(p.stdout.strip())


def measure(method: str, film: bytes, workdir: Path, junk: int = 64,
            enc_env: dict | None = None, dec_env: dict | None = None,
            extra_enc_args: list[str] | None = None) -> dict:
    """Kompletan ciklus: original -> zapis -> nezavisna rekonstrukcija -> poređenje."""
    fp, rp, op = workdir / "film.bin", workdir / "rec.bin", workdir / "out.bin"
    for p in (rp, op):
        if p.exists():
            p.unlink()
    fp.write_bytes(film)
    enc = _run([str(ROOT / "tools/encode_one.py"), method, str(fp), str(rp), *(extra_enc_args or [])], enc_env)
    row = {"method": method, "orig_bytes": len(film), "orig_sha256": hashlib.sha256(film).hexdigest(),
           "enc_ok": enc.get("ok", False), "enc_time_s": enc.get("time_s"), "enc_rss_mb": enc.get("maxrss_mb")}
    if not row["enc_ok"]:
        row.update(size=None, exact=False, note=enc.get("error", "metoda nije primenljiva"))
        return row
    row["size"] = enc["size"]
    dec = _run([str(ROOT / "tools/decode_one.py"), str(rp), str(op), "--junk", str(junk)], dec_env)
    row.update(dec_ok=dec.get("ok", False), dec_time_s=dec.get("time_s"), dec_rss_mb=dec.get("maxrss_mb"),
               model_id_at_decode=dec.get("model_id"))
    if dec.get("ok"):
        out = op.read_bytes()
        row["self_delimiting"] = dec["consumed"] == dec["rec_len"] == row["size"]
        row["out_sha256"] = hashlib.sha256(out).hexdigest()
        row["exact"] = (out == film) and row["out_sha256"] == row["orig_sha256"]
        row["byte_mismatches"] = None if len(out) != len(film) else int(
            sum(a != b for a, b in zip(out, film)) if out != film else 0)
    else:
        row.update(exact=False, note=dec.get("error"))
    for p in (fp, rp, op):
        if p.exists():
            p.unlink()
    return row


def new_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="igc_", dir=os.environ.get("IGC_TMP")))
