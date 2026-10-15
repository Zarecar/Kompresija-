"""Izveštaj KFB: spaja rezultate sa ground truth-om i računa gap prema K̄."""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path


def _f(x, nd=0):
    if x is None:
        return "—"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def load_truth(truth_dir: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in Path(truth_dir).glob("*.json")}


def summarize(res: dict, truth_dir: Path) -> str:
    truth = load_truth(truth_dir)
    rows = [r for r in res["rows"] if r["iid"] in truth]
    ok = [r for r in rows if r.get("exact") and r.get("self_delimiting") and r.get("model_valid")]
    L = ["# KFB v0 — rezultati\n",
         f"MODEL_ID (pre-registrovan): `{res['model_id_preregistered'][:16]}…`; "
         f"suite napravljen sa `{(res.get('suite_model_id') or '?')[:16]}…`. "
         f"Zapisa: {len(rows)}, tačnih i važećih: {len(ok)}. Vreme: {_f(res['wall_time_s'] / 60, 1)} min.\n",
         "K̄ = dužina oracle zapisa (proverena gornja granica složenosti u odnosu na mašinu M). "
         "Odnos = L / K̄ (geometrijska sredina po instancama); „≤K̄\" = broj instanci gde je enkoder "
         "našao zapis ne duži od oracle-a.\n"]
    # --- po instanci najbolji po klasi enkodera
    by_inst = {}
    for r in ok:
        by_inst.setdefault(r["iid"], []).append(r)

    def best(rs, pred):
        c = [r for r in rs if pred(r)]
        return min(c, key=lambda r: r["size"]) if c else None

    groups = {}
    for iid, t in truth.items():
        groups.setdefault((t["family"], t["level"], t["tier"]), []).append(iid)
    L.append("## Po porodici, nivou i veličini\n")
    L.append("| porodica | nivo | tier | n | N (B) | K̄ (B) | najbolji generički | L/K̄ | najbolji family-aware | L/K̄ | "
             "spoljni (LLM) | L/K̄ | najbolji ukupno (enkoder) | ≤K̄ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    violations = []
    for (fam, lvl, tier), iids in sorted(groups.items()):
        Kb = [truth[i]["K_bar_bytes"] for i in iids]
        Nn = [truth[i]["N_bytes"] for i in iids]
        cells = []
        for pred in (lambda r: not r["family_aware"] and not r.get("external"),
                     lambda r: r["family_aware"],
                     lambda r: r.get("external")):
            bs = [(best(by_inst.get(i, []), pred), truth[i]["K_bar_bytes"]) for i in iids]
            have = [(b, k) for b, k in bs if b is not None]
            if not have:
                cells += ["—", "—"]
                continue
            gm = math.exp(st.mean(math.log(b["size"] / k) for b, k in have))
            cells += [f"{_f(st.mean(b['size'] for b, _ in have))} ({len(have)}/{len(iids)})", _f(gm, 2)]
        overall = [best(by_inst.get(i, []), lambda r: True) for i in iids]
        names = sorted({b["encoder"] for b in overall if b})
        beat = sum(1 for b, i in zip(overall, iids) if b and b["size"] <= truth[i]["K_bar_bytes"])
        L.append(f"| {fam} | {lvl} | {tier} | {len(iids)} | {_f(st.mean(Nn))} | {_f(st.mean(Kb))} | "
                 + " | ".join(cells) + f" | {', '.join(names) or '—'} | {beat}/{len(iids)} |")
        if fam == "control":
            for i in iids:
                payload_len = truth[i]["payload_bytes"]
                for r in by_inst.get(i, []):
                    if r["size"] < payload_len:
                        violations.append((i, r["encoder"], r["size"], payload_len))
    L.append("\n## Kontrole\n")
    L.append(f"- CSPRNG kontrola: {'**PREKRŠENA** ' + str(violations) if violations else 'nijedan tačan zapis kraći od sadržaja (OK)'}.")
    bad_model = [r for r in rows if r.get("dec_ok") and not r.get("model_valid")]
    L.append(f"- Zapisi dekodovani sa drugim MODEL_ID: {len(bad_model)}.")
    not_sd = [r for r in rows if r.get("dec_ok") and r.get("exact") and not r.get("self_delimiting")]
    L.append(f"- Tačni ali ne samoograničavajući (nevažeći): {len(not_sd)}.")
    inexact = [r for r in rows if r.get("dec_ok") and not r.get("exact")]
    L.append(f"- Dekodovani ali NETAČNI zapisi: {len(inexact)} " +
             (f"({', '.join(sorted({r['encoder'] for r in inexact}))})" if inexact else ""))
    L.append("\n## Po enkoderu\n")
    L.append("| enkoder | family-aware | pokušaja | tačnih | najbolji na instanci | sr. enc s | sr. dec s | max enc RSS MB |")
    L.append("|---|---|---|---|---|---|---|---|")
    winners = {}
    for iid, rs in by_inst.items():
        b = min(rs, key=lambda r: r["size"])
        winners[b["encoder"]] = winners.get(b["encoder"], 0) + 1
    for name in sorted({r["encoder"] for r in rows}):
        rr = [r for r in rows if r["encoder"] == name]
        ex = [r for r in rr if r in ok]
        et = [r["enc_time_s"] for r in rr if r.get("enc_time_s") is not None]
        dt = [r["dec_time_s"] for r in ex if r.get("dec_time_s") is not None]
        rs_ = [r["enc_rss_mb"] for r in rr if r.get("enc_rss_mb")]
        L.append(f"| {name} | {'da' if rr[0].get('family_aware') else 'ne'} | {len(rr)} | {len(ex)} | "
                 f"{winners.get(name, 0)} | {_f(st.mean(et), 2) if et else '—'} | {_f(st.mean(dt), 2) if dt else '—'} | "
                 f"{_f(max(rs_)) if rs_ else '—'} |")
    return "\n".join(L) + "\n"
