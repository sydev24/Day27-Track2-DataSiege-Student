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


def check_data_batch(payload, ctx):
    res = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="checks")
    
    b = ctx.baseline
    reasons = []
    if res["row_count"] < b["row_count_min"] or res["row_count"] > b["row_count_max"]:
        reasons.append(f"row_count out of bounds: {res['row_count']}")
    if res["null_rate"].get("customer_id", 0) > b["null_rate_max"]:
        reasons.append(f"null_rate too high: {res['null_rate'].get('customer_id')}")
    if res["mean_amount"] < b["mean_amount_min"] or res["mean_amount"] > b["mean_amount_max"]:
        reasons.append(f"mean_amount out of bounds: {res['mean_amount']}")
    if res["staleness_min"] > b["staleness_min_max"]:
        reasons.append(f"staleness too high: {res['staleness_min']}")
        
    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="checks")
    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    res = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="contracts")
    
    b = ctx.baseline
    reasons = []
    if res.get("violations"):
        reasons.append(f"violations: {res['violations']}")
    if res.get("freshness_delay_min", 0) > b["freshness_delay_max_min"]:
        reasons.append(f"freshness delay too high: {res['freshness_delay_min']}")
        
    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="contracts")
    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    res = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="lineage")
    
    b = ctx.baseline
    reasons = []
    if res.get("duration_ms", 0) > b["lineage_duration_ms_max"]:
        reasons.append(f"duration too high: {res['duration_ms']}")
        
    if res.get("actual_downstream_count", 1) == 0:
        reasons.append("orphan output detected (downstream count is 0)")
        
    job = payload.get("job", "default_job")
    upstreams = ctx.state.setdefault("lineage_upstreams", {})
    actual_up = set(res.get("actual_upstream", []))
    if job in upstreams:
        if actual_up < upstreams[job] or len(actual_up) < len(upstreams[job]):
            reasons.append(f"missing upstream for {job}: expected {upstreams[job]}, got {actual_up}")
        else:
            upstreams[job].update(actual_up)
    else:
        upstreams[job] = actual_up
    
    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="lineage")
    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    res = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra")
    
    b = ctx.baseline
    if res.get("mean_shift_sigma", 0) > b["feature_mean_shift_sigma_max"]:
        return Verdict(alert=True, reason=f"feature drift: {res['mean_shift_sigma']}", pillar="ai_infra")
    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    res = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra")
    
    b = ctx.baseline
    reasons = []
    if res.get("centroid_shift", 0) > b["embedding_centroid_shift_max"]:
        reasons.append(f"centroid shift too high: {res['centroid_shift']}")
    if res.get("avg_doc_age_days", 0) > b["corpus_avg_doc_age_days_max"]:
        reasons.append(f"doc age too high: {res['avg_doc_age_days']}")
        
    if reasons:
        return Verdict(alert=True, reason="; ".join(reasons), pillar="ai_infra")
    return Verdict(alert=False, pillar="ai_infra")
