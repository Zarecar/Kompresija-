"""Pokretač KFB: svaki enkoder i dekoder u zasebnom procesu, poređenje bajt po bajt.

    python3 -m kfb.run --suite DIR --out results/kfb [--encoders all] [--jobs 3]
                       [--budget-mini 20] [--budget-std 60] [--extra-records DIR]

--extra-records DIR: već gotovi zapisi spoljnih enkodera (npr. LLM agenata),
    imenovani <iid>__<ime>.rec (opciono <iid>__<ime>.json sa {"time_s":..}).
    Verifikuju se ISTIM nezavisnim dekoderom.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kfb import encoders, record  # noqa: E402

PY = sys.executable


def _sub(args: list[str], timeout: float) -> dict:
    try:
        p = subprocess.run([PY, "-m", *args], capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if p.returncode != 0:
        return {"ok": False, "error": (p.stderr.strip() or "greška")[-400:]}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": "neispravan izlaz: " + p.stdout[-200:]}


def verify(rec_path: Path, film: bytes, work: Path) -> dict:
    out = work / "out.film"
    if out.exists():
        out.unlink()
    d = _sub(["kfb.tools.decode", str(rec_path), str(out), "--junk", "64"], timeout=900)
    row = {"dec_ok": d.get("ok", False), "dec_time_s": d.get("time_s"), "dec_rss_mb": d.get("rss_mb"),
           "model_id_at_decode": d.get("model_id")}
    if d.get("ok"):
        got = out.read_bytes()
        row["exact"] = got == film
        row["self_delimiting"] = d["consumed"] == d["rec_len"]
        row["out_sha256"] = hashlib.sha256(got).hexdigest()
        out.unlink()
    else:
        row["exact"] = False
        row["error"] = d.get("error")
    return row


def job(iid: str, tier: str, film_path: Path, enc_name: str, budget: float, tmp: Path) -> dict:
    work = Path(tempfile.mkdtemp(dir=tmp))
    film = film_path.read_bytes()
    rec = work / "rec.bin"
    e = _sub(["kfb.tools.encode", enc_name, str(film_path), str(rec), str(budget)], timeout=2 * budget + 60)
    row = {"iid": iid, "tier": tier, "encoder": enc_name, "budget_s": budget,
           "enc_ok": e.get("ok", False), "enc_time_s": e.get("time_s"), "enc_rss_mb": e.get("rss_mb")}
    if not e.get("ok"):
        row.update(size=None, exact=False, note=e.get("error", "nije primenljiv"))
    else:
        row["size"] = e["size"]
        row.update(verify(rec, film, work))
    for p in work.iterdir():
        p.unlink()
    work.rmdir()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", default="all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--budget-mini", type=float, default=20)
    ap.add_argument("--budget-std", type=float, default=60)
    ap.add_argument("--extra-records", default="")
    ap.add_argument("--only-extra", action="store_true")
    a = ap.parse_args()
    suite = Path(a.suite)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mid = record.model_id()
    man = json.loads((suite / "manifest.json").read_text())
    encs = encoders.load_all()
    names = list(encs) if a.encoders == "all" else a.encoders.split(",")
    budgets = {"mini": a.budget_mini, "std": a.budget_std}
    tmp = Path(tempfile.mkdtemp(prefix="kfb_", dir=os.environ.get("IGC_TMP")))
    jobs = []
    if not a.only_extra:
        for inst in man["instances"]:
            for n in names:
                if inst["tier"] in encs[n].tiers:
                    jobs.append((inst["iid"], inst["tier"], suite / "films" / f"{inst['iid']}.film", n,
                                 budgets[inst["tier"]]))
    t0 = time.time()
    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(job, *j, tmp): j for j in jobs}
        for i, f in enumerate(cf.as_completed(futs)):
            r = f.result()
            r["model_valid"] = r.get("model_id_at_decode") in (None, mid) if not r.get("dec_ok") else \
                r.get("model_id_at_decode") == mid
            r["family_aware"] = encs[r["encoder"]].family_aware
            rows.append(r)
            print(f"[{time.time()-t0:7.0f}s {i+1}/{len(jobs)}] {r['iid']} {r['tier']:4s} {r['encoder']:14s} "
                  f"size={r.get('size')} exact={r.get('exact')}", flush=True)
    if a.extra_records:
        tiers = {i["iid"]: i["tier"] for i in man["instances"]}
        for rp in sorted(Path(a.extra_records).glob("*.rec")):
            iid, name = rp.stem.split("__", 1)
            if iid not in tiers:
                continue
            side = rp.with_suffix(".json")
            meta = json.loads(side.read_text()) if side.exists() else {}
            work = Path(tempfile.mkdtemp(dir=tmp))
            film = (suite / "films" / f"{iid}.film").read_bytes()
            r = {"iid": iid, "tier": tiers[iid], "encoder": f"ext:{name}", "budget_s": None,
                 "enc_ok": True, "enc_time_s": meta.get("time_s"), "enc_rss_mb": None,
                 "size": rp.stat().st_size, "family_aware": False, "external": True,
                 "external_meta": meta}
            r.update(verify(rp, film, work))
            r["model_valid"] = r.get("model_id_at_decode") == mid
            work.rmdir()
            rows.append(r)
            print(f"[ext] {iid} {name} size={r['size']} exact={r['exact']}", flush=True)
    res = {"model_id_preregistered": mid, "suite_model_id": man.get("model_id"), "rows": rows,
           "wall_time_s": time.time() - t0, "budgets": budgets, "jobs": a.jobs}
    (out / "rows.json").write_text(json.dumps(res, indent=1))
    from kfb import report
    (out / "summary.md").write_text(report.summarize(res, suite / "truth"))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
