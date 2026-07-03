"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


# ── Online statistics (Welford's algorithm) ───────────────────────

def _push(tk, key, val):
    """Add value to running accumulator for key."""
    if key not in tk:
        tk[key] = {"n": 0, "mean": 0.0, "M2": 0.0}
    s = tk[key]
    s["n"] += 1
    d1 = val - s["mean"]
    s["mean"] += d1 / s["n"]
    d2 = val - s["mean"]
    s["M2"] += d1 * d2


def _mu_sig(tk, key, min_n=5):
    """Return (mean, stdev) or (None, None) if fewer than min_n samples."""
    if key not in tk or tk[key]["n"] < min_n:
        return None, None
    s = tk[key]
    var = s["M2"] / max(s["n"] - 1, 1)
    return s["mean"], max(var ** 0.5, 1e-9)


def _absz(val, mu, sig):
    """Absolute z-score; 0 when stats unavailable."""
    if mu is None or sig is None or sig < 1e-12:
        return 0.0
    return abs(val - mu) / sig


def _dirz(val, mu, sig):
    """Signed z-score; positive = above mean."""
    if mu is None or sig is None or sig < 1e-12:
        return 0.0
    return (val - mu) / sig


# ── data_batch ────────────────────────────────────────────────────

def check_data_batch(payload, ctx):
    res = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="checks")

    b = ctx.baseline
    reasons = []

    rc  = res["row_count"]
    nr  = res["null_rate"].get("customer_id", 0)
    ma  = res["mean_amount"]
    sa  = res.get("std_amount", 0)
    stl = res["staleness_min"]

    # ── Hard baseline boundary checks ──
    if rc < b["row_count_min"] or rc > b["row_count_max"]:
        reasons.append(f"row_count OOB: {rc}")
    if nr > b["null_rate_max"]:
        reasons.append(f"null_rate high: {nr}")
    if ma < b["mean_amount_min"] or ma > b["mean_amount_max"]:
        reasons.append(f"mean_amount OOB: {ma}")
    if stl > b["staleness_min_max"]:
        reasons.append(f"staleness high: {stl}")

    # ── Adaptive z-score detection ──
    tk = ctx.state.setdefault("db_tk", {})

    if not reasons:
        zs = {}
        for k, v in [("rc", rc), ("ma", ma), ("nr", nr), ("sa", sa), ("stl", stl)]:
            z = _absz(v, *_mu_sig(tk, k, min_n=8))
            if z > 0:
                zs[k] = z

        if zs:
            max_z = max(zs.values())
            n_hi = sum(1 for v in zs.values() if v > 2.5)
            n_md = sum(1 for v in zs.values() if v > 2.0)

            if max_z > 3.0:
                reasons.append(f"z-anomaly: {max_z:.1f}")
            elif n_hi >= 1 and n_md >= 3:
                reasons.append(f"multi-z: {n_hi}h/{n_md}m")
            elif n_md >= 4:
                reasons.append(f"broad-z: {n_md} elevated")

    # Only update tracker for events we consider clean
    if not reasons:
        for k, v in [("rc", rc), ("ma", ma), ("nr", nr), ("sa", sa), ("stl", stl)]:
            _push(tk, k, v)

    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="checks")
    return Verdict(alert=False, pillar="checks")


# ── contract_checkpoint ───────────────────────────────────────────

def check_contract_checkpoint(payload, ctx):
    res = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="contracts")

    b = ctx.baseline
    reasons = []
    if res.get("violations"):
        reasons.append(f"violations: {res['violations']}")
    if res.get("freshness_delay_min", 0) > b["freshness_delay_max_min"]:
        reasons.append(f"freshness high: {res['freshness_delay_min']}")

    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="contracts")
    return Verdict(alert=False, pillar="contracts")


# ── lineage_run ───────────────────────────────────────────────────

def check_lineage_run(payload, ctx):
    res = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="lineage")

    b = ctx.baseline
    reasons = []
    dur = res.get("duration_ms", 0)

    # Hard baseline
    if dur > b["lineage_duration_ms_max"]:
        reasons.append(f"duration high: {dur}")

    # Orphan output
    if res.get("actual_downstream_count", 1) == 0:
        reasons.append("orphan output")

    # Upstream topology check with progressive learning
    job = payload.get("job", "default_job")
    ups = ctx.state.setdefault("lr_ups", {})
    actual = set(res.get("actual_upstream", []))

    if job in ups:
        expected = ups[job]
        if not actual.issuperset(expected):
            reasons.append(f"missing upstream for {job}")
        elif actual > expected:
            ups[job] = actual
    else:
        ups[job] = actual

    # Adaptive duration z-score
    tk = ctx.state.setdefault("lr_tk", {})
    z_dur = _absz(dur, *_mu_sig(tk, "dur", min_n=5))
    if not reasons and z_dur > 2.5:
        reasons.append(f"duration z={z_dur:.1f}")

    if not reasons:
        _push(tk, "dur", dur)

    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="lineage")
    return Verdict(alert=False, pillar="lineage")


# ── feature_materialization ───────────────────────────────────────

def check_feature_materialization(payload, ctx):
    res = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra")

    b = ctx.baseline
    if res.get("mean_shift_sigma", 0) > b["feature_mean_shift_sigma_max"]:
        return Verdict(alert=True,
                       reason=f"feature drift: {res['mean_shift_sigma']}",
                       pillar="ai_infra")
    return Verdict(alert=False, pillar="ai_infra")


# ── embedding_batch ───────────────────────────────────────────────

def check_embedding_batch(payload, ctx):
    res = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra")

    b = ctx.baseline
    reasons = []

    cs = res.get("centroid_shift", 0)
    da = res.get("avg_doc_age_days", 0)

    # Hard baseline checks
    if cs > b["embedding_centroid_shift_max"]:
        reasons.append(f"centroid shift high: {cs}")
    if da > b["corpus_avg_doc_age_days_max"]:
        reasons.append(f"doc age high: {da}")

    # ── Adaptive pattern detection ──
    tk = ctx.state.setdefault("eb_tk", {})
    mu_cs, sig_cs = _mu_sig(tk, "cs", min_n=5)
    mu_da, sig_da = _mu_sig(tk, "da", min_n=5)

    if not reasons and mu_cs is not None and mu_da is not None:
        dz_da = _dirz(da, mu_da, sig_da)   # positive = old corpus
        dz_cs = _dirz(cs, mu_cs, sig_cs)   # negative = no shift

        # Corpus staleness pattern: old docs + stable embeddings
        if dz_da > 1.0 and dz_cs < -0.5:
            combo = dz_da - dz_cs
            if combo > 2.5:
                reasons.append(f"staleness: da_z={dz_da:.1f} cs_z={dz_cs:.1f}")

        # General outlier
        if not reasons:
            az_da = _absz(da, mu_da, sig_da)
            az_cs = _absz(cs, mu_cs, sig_cs)
            if az_da > 3.0 or az_cs > 3.0:
                reasons.append(f"outlier: da_z={az_da:.1f} cs_z={az_cs:.1f}")

    # Only update tracker for events we consider clean
    if not reasons:
        _push(tk, "cs", cs)
        _push(tk, "da", da)

    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="ai_infra")
    return Verdict(alert=False, pillar="ai_infra")
