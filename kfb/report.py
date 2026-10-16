"""Izveštaj KFB: spaja rezultate sa ground truth-om i računa gap prema K̄.

Klase rezultata (po tipu zapisa i enkoderu, da se rezervni zapisi ne mešaju):
  generički   najbolji tačan zapis koji NIJE KFL program (raw, zlib, lzma, bz2, cm,
              mt_infer, ffmpeg...), bez obzira koji ga je enkoder vratio
  KFL-porodica najbolji tačan KFL zapis family-aware enkodera
  KFL-pretraga najbolji tačan KFL zapis generičkog (ne family-aware) enkodera
  LLM          najbolji tačan zapis spoljnog enkodera (sandbox)
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

KFL_METHOD = 0x10
CLASSES = ["generički", "KFL-porodica", "KFL-pretraga", "LLM"]


def _f(x, nd=0):
    if x is None:
        return "—"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def load_truth(truth_dir: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in Path(truth_dir).glob("*.json")}


def klass(r: dict) -> str:
    if r.get("external"):
        return "LLM"
    if r.get("method") != KFL_METHOD:
        return "generički"
    return "KFL-porodica" if r.get("family_aware") else "KFL-pretraga"


def method_name(m) -> str:
    if m == KFL_METHOD:
        return "kfl"
    try:
        from igc import codec as ic
        if m in ic.NAMES:
            return ic.NAMES[m]
        from kfb import ffcodec
        inv = {v: k for k, v in getattr(ffcodec, "METHODS", {}).items()}
        return inv.get(m, f"0x{m:02x}")
    except Exception:  # izveštaj ne sme da padne zbog prikaza
        return f"0x{m:02x}" if isinstance(m, int) else "?"


def label(r: dict) -> str:
    return r["encoder"] if r.get("method") == KFL_METHOD else f"{r['encoder']}→{method_name(r.get('method'))}"


def _gm(xs):
    return math.exp(st.mean(math.log(x) for x in xs)) if xs else None


def summarize(res: dict, truth_dir: Path) -> str:
    truth = load_truth(truth_dir)
    rows = [r for r in res["rows"] if r["iid"] in truth]
    ok = [r for r in rows if r.get("exact") and r.get("self_delimiting") and r.get("model_valid")]
    by_inst: dict = {}
    for r in ok:
        by_inst.setdefault(r["iid"], []).append(r)

    def best(iid, cls=None):
        c = [r for r in by_inst.get(iid, []) if cls is None or klass(r) == cls]
        return min(c, key=lambda r: r["size"]) if c else None

    L = ["# KFB v0: rezultati\n",
         f"MODEL_ID (pre-registrovan): `{res['model_id_preregistered'][:16]}…`; skup napravljen sa "
         f"`{(res.get('suite_model_id') or '?')[:16]}…`. Zapisa: {len(rows)}, tačnih i važećih: {len(ok)}. "
         f"Ukupno vreme: {_f(res['wall_time_s'] / 60, 1)} min (paralelno {res.get('jobs')} procesa).\n",
         "K̄ je dužina oracle zapisa (proverena gornja granica složenosti u odnosu na mašinu M). L/K̄ je "
         "geometrijska sredina po instancama u grupi; ≤K̄ je broj instanci gde je najbolji zapis klase ne "
         "duži od oracle-a. Klase su definisane u zaglavlju `kfb/report.py`.\n"]
    groups: dict = {}
    for iid, t in truth.items():
        groups.setdefault((t["family"], t["level"], t["tier"]), []).append(iid)

    # ---------------------------------------------- ukupni pregled po klasi
    L.append("## Ukupno po klasi enkodera (sve instance osim CSPRNG kontrole)\n")
    L.append("| klasa | instanci sa tačnim zapisom | L/K̄ (geom.) | ≤ K̄ | ≤ 1,1·K̄ | ≤ 1 KB |")
    L.append("|---|---|---|---|---|---|")
    nonctl = [i for i, t in truth.items() if t["family"] != "control"]
    for cls in CLASSES + [None]:
        bs = [(best(i, cls), truth[i]) for i in nonctl]
        have = [(b, t) for b, t in bs if b]
        if not have:
            continue
        ratios = [b["size"] / t["K_bar_bytes"] for b, t in have]
        name = cls or "**najbolji ukupno**"
        L.append(f"| {name} | {len(have)}/{len(nonctl)} | {_f(_gm(ratios), 2)} | "
                 f"{sum(r <= 1 for r in ratios)} | {sum(r <= 1.1 for r in ratios)} | "
                 f"{sum(b['size'] <= 1024 for b, _ in have)} |")
    kbar_1k = sum(truth[i]["K_bar_bytes"] <= 1024 for i in nonctl)
    L.append(f"\nOracle (K̄) staje u 1 KB na {kbar_1k}/{len(nonctl)} instanci.\n")

    # ---------------------------------------------- po porodici
    L.append("## Po porodici, nivou i veličini\n")
    L.append("| porodica | nivo | tier | n | N (B) | K̄ (B) | " +
             " | ".join(f"{c} L (L/K̄)" for c in CLASSES) + " | najbolji (enkoder) | ≤K̄ |")
    L.append("|---|---|---|---|---|---|" + "---|" * len(CLASSES) + "---|---|")
    violations = []
    for (fam, lvl, tier), iids in sorted(groups.items()):
        Kb = [truth[i]["K_bar_bytes"] for i in iids]
        Nn = [truth[i]["N_bytes"] for i in iids]
        cells = []
        for cls in CLASSES:
            have = [(best(i, cls), truth[i]["K_bar_bytes"]) for i in iids]
            have = [(b, k) for b, k in have if b]
            if not have:
                cells.append("—")
                continue
            cells.append(f"{_f(st.mean(b['size'] for b, _ in have))} ({_f(_gm([b['size'] / k for b, k in have]), 2)})"
                         + ("" if len(have) == len(iids) else f" [{len(have)}/{len(iids)}]"))
        overall = [best(i) for i in iids]
        names = sorted({label(b) for b in overall if b})
        beat = sum(1 for b, i in zip(overall, iids) if b and b["size"] <= truth[i]["K_bar_bytes"])
        L.append(f"| {fam} | {lvl} | {tier} | {len(iids)} | {_f(st.mean(Nn))} | {_f(st.mean(Kb))} | "
                 + " | ".join(cells) + f" | {', '.join(names) or '—'} | {beat}/{len(iids)} |")
        if fam == "control":
            for i in iids:
                for r in by_inst.get(i, []):
                    if r["size"] < truth[i]["payload_bytes"]:
                        violations.append((i, r["encoder"], r["size"], truth[i]["payload_bytes"]))

    # ---------------------------------------------- kontrole
    L.append("\n## Kontrole\n")
    L.append("- CSPRNG kontrola: " + ("**PREKRŠENA** " + str(violations) if violations else
                                     "nijedan tačan zapis nije kraći od sadržaja (OK)") + ".")
    bad_model = [r for r in rows if r.get("dec_ok") and not r.get("model_valid")]
    L.append(f"- Zapisi dekodovani sa drugim MODEL_ID: {len(bad_model)}.")
    not_sd = [r for r in rows if r.get("dec_ok") and r.get("exact") and not r.get("self_delimiting")]
    L.append(f"- Tačni, ali ne samoograničavajući (nevažeći): {len(not_sd)}.")
    inexact = [r for r in rows if r.get("dec_ok") and not r.get("exact")]
    L.append(f"- Dekodovani, ali NETAČNI zapisi: {len(inexact)}"
             + (f" ({', '.join(sorted({r['encoder'] for r in inexact}))})" if inexact else "") + ".")
    failed_dec = [r for r in rows if r.get("enc_ok") and not r.get("dec_ok")]
    L.append(f"- Zapisi koje dekoder nije mogao da pročita: {len(failed_dec)}"
             + (f" ({', '.join(sorted({r['encoder'] for r in failed_dec}))})" if failed_dec else "") + ".")
    timeouts = [r for r in rows if r.get("note") == "timeout"]
    L.append(f"- Prekoračenje vremena enkodera: {len(timeouts)}"
             + (f" ({', '.join(sorted({r['encoder'] for r in timeouts}))})" if timeouts else "") + ".")

    # ---------------------------------------------- po enkoderu
    L.append("\n## Po enkoderu\n")
    L.append("| enkoder | family-aware | pokušaja | tačnih | od toga KFL | najbolji na instanci | sr. enc s | max enc s | sr. dec s | max enc RSS MB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    winners: dict = {}
    for iid in by_inst:
        b = best(iid)
        winners[b["encoder"]] = winners.get(b["encoder"], 0) + 1
    for name in sorted({r["encoder"] for r in rows}):
        rr = [r for r in rows if r["encoder"] == name]
        ex = [r for r in rr if r in ok]
        et = [r["enc_time_s"] for r in rr if r.get("enc_time_s") is not None]
        dt = [r["dec_time_s"] for r in ex if r.get("dec_time_s") is not None]
        rs_ = [r["enc_rss_mb"] for r in rr if r.get("enc_rss_mb")]
        L.append(f"| {name} | {'da' if rr[0].get('family_aware') else 'ne'} | {len(rr)} | {len(ex)} | "
                 f"{sum(1 for r in ex if r.get('method') == KFL_METHOD)} | {winners.get(name, 0)} | "
                 f"{_f(st.mean(et), 2) if et else '—'} | {_f(max(et), 1) if et else '—'} | "
                 f"{_f(st.mean(dt), 2) if dt else '—'} | {_f(max(rs_)) if rs_ else '—'} |")
    return "\n".join(L) + "\n"
