"""
test_cache_interceptor.py  — AIM Plant Advisor
================================================
Standalone test runner for PreCacheInterceptor (no-bypass version).

Every query — regardless of severity, urgency, count tier, or compound nature —
always goes through the cache lookup path. No bypasses.

Tests ALL six similarity-trap patterns:
  1. Numeric range     — same structure, different magnitude → different cache key
  2. Temporal urgency  — historical vs active_now → different cache keys
  3. Named entity      — AIM-5000 vs AIM-10000, Zone 1 vs Zone 3, alloys
  4. Negation          — affirmative vs negated → different keys
  5. Count / ordinal   — 1st vs 2nd vs 3+ → different keys (all cached)
  6. Compound condition — multi-fault queries → own cache entry

Cache behaviour tested:
  - MISS  → calls RAG stub, stores answer, returns it
  - HIT   → returns cached answer (same signal group, different phrasing)
  - ISOLATION → different signal groups must NOT share a key

Run:
  python test_cache_interceptor.py
  python test_cache_interceptor.py --verbose
  python test_cache_interceptor.py --pattern numeric
  python test_cache_interceptor.py --query "H2 is 0.22 ml/100g — what do I do?"
"""

import sys
import time
import argparse
from collections import defaultdict

from pre_cache_interceptor import PreCacheInterceptor


# ─────────────────────────────────────────────────────────────────────────────
# Test case schema
# ─────────────────────────────────────────────────────────────────────────────

class TC:
    def __init__(
        self,
        query:             str,
        pattern:           str,
        group:             str,
        expect_hit_after:  str  = "",   # group that must be cached before this can HIT
        expect_range_tag:  str  = "",
        expect_negation:   str  = "",
        expect_count_tier: str  = "",
        expect_entity:     str  = "",
        expect_compound:   bool = False,
        desc:              str  = "",
    ):
        self.query             = query
        self.pattern           = pattern
        self.group             = group
        self.expect_hit_after  = expect_hit_after
        self.expect_range_tag  = expect_range_tag
        self.expect_negation   = expect_negation
        self.expect_count_tier = expect_count_tier
        self.expect_entity     = expect_entity
        self.expect_compound   = expect_compound
        self.desc              = desc


# ─────────────────────────────────────────────────────────────────────────────
# Test suite — 47 cases, bypass expectations removed
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES: list[TC] = [

    # ── Pattern 1: Numeric range ──────────────────────────────────────────────

    TC("H2 level is 0.08 ml/100g Al — is this acceptable for casting?",
       pattern="numeric", group="h2_normal",
       expect_range_tag="h2_level:normal",
       desc="H2 normal zone — first query, must store"),

    TC("Hydrogen content reading shows 0.09 ml/100g — are we OK to proceed?",
       pattern="numeric", group="h2_normal",
       expect_range_tag="h2_level:normal", expect_hit_after="h2_normal",
       desc="H2 normal zone — paraphrase must HIT"),

    TC("H2 is 0.07 ml/100g Al — good to cast?",
       pattern="numeric", group="h2_normal",
       expect_range_tag="h2_level:normal", expect_hit_after="h2_normal",
       desc="H2 normal zone — third variant must HIT"),

    TC("H2 level is 0.17 ml/100g — should I cast?",
       pattern="numeric", group="h2_reject",
       expect_range_tag="h2_level:reject",
       desc="H2 reject zone — must NOT hit h2_normal cache"),

    TC("Hydrogen content at 0.19 ml/100g Al — continue or stop?",
       pattern="numeric", group="h2_reject",
       expect_range_tag="h2_level:reject", expect_hit_after="h2_reject",
       desc="H2 reject zone paraphrase — must HIT h2_reject cache"),

    TC("H2 is 0.22 ml/100g — what do I do?",
       pattern="numeric", group="h2_dump",
       expect_range_tag="h2_level:dump",
       desc="H2 dump zone — now cached like any other range"),

    TC("Compressor temperature showing 45.4 deg C — is this bad?",
       pattern="numeric", group="comp_temp_normal",
       expect_range_tag="compressor_temp:normal",
       desc="Compressor normal zone — first query"),

    TC("Compressor temp is 44.9 or 45.8 degC — what should I do?",
       pattern="numeric", group="comp_temp_normal",
       expect_range_tag="compressor_temp:normal", expect_hit_after="comp_temp_normal",
       desc="Compressor normal zone — dual-value query must HIT"),

    TC("Compressor temperature is showing 65.8 deg C — is this bad?",
       pattern="numeric", group="comp_temp_critical",
       expect_range_tag="compressor_temp:critical",
       desc="Compressor CRITICAL zone — different key from normal"),

    TC("My compressor temp is 72 deg C — what now?",
       pattern="numeric", group="comp_temp_danger",
       expect_range_tag="compressor_temp:danger",
       desc="Compressor danger zone — now cached"),

    TC("Metal temp is 720°C — is that fine for 6063 alloy?",
       pattern="numeric", group="metal_temp_normal",
       expect_range_tag="metal_temp:normal", expect_entity="alloy:6063",
       desc="Metal temp normal + alloy entity"),

    TC("Cooling water flow is at 85% — do I need to adjust?",
       pattern="numeric", group="cw_normal",
       expect_range_tag="cooling_water_flow:normal",
       desc="Cooling water normal zone"),

    TC("Water flow reading is 55% — F-021 risk?",
       pattern="numeric", group="cw_trip",
       expect_range_tag="cooling_water_flow:trip",
       desc="Cooling water trip zone — now cached"),

    TC("ILD gas flow is 10 Nl/min — is that acceptable?",
       pattern="numeric", group="ild_normal",
       expect_range_tag="ild_gas_flow:normal",
       desc="ILD gas flow normal zone"),

    TC("ILD nitrogen flow dropped to 4 Nl/min — what now?",
       pattern="numeric", group="ild_trip",
       expect_range_tag="ild_gas_flow:trip",
       desc="ILD gas flow trip zone — now cached"),

    # ── Pattern 2: Temporal urgency ───────────────────────────────────────────

    TC("The cooling water tripped last Tuesday — what was the cause?",
       pattern="temporal", group="cw_historical",
       desc="Historical event — cacheable"),

    TC("Cooling water flow tripped yesterday — probable reasons?",
       pattern="temporal", group="cw_historical",
       expect_hit_after="cw_historical",
       desc="Historical paraphrase — must HIT"),

    TC("Cooling water is tripping RIGHT NOW — what do I do immediately?",
       pattern="temporal", group="cw_active_now",
       desc="Active-now query — gets its own cache entry (temporal:active_now key)"),

    TC("Burner 2 failed to ignite last Monday — root cause?",
       pattern="temporal", group="burner_historical",
       desc="Historical burner fault — cacheable"),

    TC("Burner 2 is failing to ignite right now — what do I do?",
       pattern="temporal", group="burner_active_now",
       desc="Active-now burner fault — own cache entry"),

    TC("H2 was high in the last heat — what caused it?",
       pattern="temporal", group="cw_historical",
       expect_hit_after="cw_historical",
       desc="Historical H2 — shares key with other historical queries"),

    TC("H2 is currently elevated — can we cast?",
       pattern="temporal", group="h2_active_now",
       desc="Active-now H2 — own cache entry"),

    # ── Pattern 3: Named entity ───────────────────────────────────────────────

    TC("What is the cooling water flow spec for the AIM-5000?",
       pattern="entity", group="cw_spec_5000",
       expect_entity="model:AIM-5000",
       desc="AIM-5000 spec — must store with model key"),

    TC("What cooling water flow is required for AIM-5000?",
       pattern="entity", group="cw_spec_5000",
       expect_entity="model:AIM-5000", expect_hit_after="cw_spec_5000",
       desc="AIM-5000 spec paraphrase — must HIT"),

    TC("What is the cooling water flow spec for the AIM-10000?",
       pattern="entity", group="cw_spec_10000",
       expect_entity="model:AIM-10000",
       desc="AIM-10000 spec — MUST NOT hit AIM-5000 cache"),

    TC("Burner 1 in zone 1 failed to ignite — what do I check?",
       pattern="entity", group="burner1_zone1",
       expect_entity="zone:1",
       desc="Zone 1 fault — entity tag in key"),

    TC("Burner 3 in zone 3 failed to ignite — what do I check?",
       pattern="entity", group="burner3_zone3",
       expect_entity="zone:3",
       desc="Zone 3 fault — MUST NOT hit zone 1 cache"),

    TC("What is the casting temperature for 5083 alloy?",
       pattern="entity", group="temp_5083",
       expect_entity="alloy:5083",
       desc="5083 alloy — entity-specific answer"),

    TC("What casting temp should I use for 6063?",
       pattern="entity", group="temp_6063",
       expect_entity="alloy:6063",
       desc="6063 alloy — MUST NOT hit 5083 cache"),

    # ── Pattern 4: Negation ───────────────────────────────────────────────────

    TC("Is it safe to add wet scrap to the furnace while casting?",
       pattern="negation", group="wet_scrap_affirm",
       expect_negation=":affirm",
       desc="Affirmative safety question — cacheable"),

    TC("Can wet scrap be charged into the furnace during a cast?",
       pattern="negation", group="wet_scrap_affirm",
       expect_negation=":affirm", expect_hit_after="wet_scrap_affirm",
       desc="Affirmative paraphrase — must HIT"),

    TC("Is it NOT safe to add wet scrap to the furnace?",
       pattern="negation", group="wet_scrap_negated",
       expect_negation=":negated",
       desc="Negated form — MUST NOT hit affirm cache"),

    TC("Should I avoid adding wet scrap to the furnace?",
       pattern="negation", group="wet_scrap_negated",
       expect_negation=":negated", expect_hit_after="wet_scrap_negated",
       desc="Negated paraphrase (avoid) — must HIT negated cache"),

    TC("Can I restart the casting table after an E-stop?",
       pattern="negation", group="estop_restart_affirm",
       expect_negation=":affirm",
       desc="Affirmative restart question"),

    TC("Should I NOT restart the casting table after E-stop?",
       pattern="negation", group="estop_restart_negated",
       expect_negation=":negated",
       desc="Negated restart — MUST NOT hit affirm cache"),

    # ── Pattern 5: Count / ordinal ────────────────────────────────────────────

    TC("Burner failed to ignite for the first time — what do I check?",
       pattern="count", group="burner_fail_1",
       expect_count_tier="count:1",
       desc="First ignition failure — cacheable"),

    TC("Burner ignition failed once — what should I inspect?",
       pattern="count", group="burner_fail_1",
       expect_count_tier="count:1", expect_hit_after="burner_fail_1",
       desc="First failure paraphrase — must HIT"),

    TC("Burner failed to ignite a second time — next steps?",
       pattern="count", group="burner_fail_2",
       expect_count_tier="count:2",
       desc="Second failure — different key from first"),

    TC("Burner ignition has failed 3 times in a row — what now?",
       pattern="count", group="burner_fail_3plus",
       expect_count_tier="count:3+",
       desc="Third failure — now cached (count:3+ is a key signal, not a bypass)"),

    TC("Burner keeps failing to ignite — what's the issue?",
       pattern="count", group="burner_fail_repeated",
       expect_count_tier="count:repeated",
       desc="Repeated pattern — separate cache entry"),

    TC("ILD rotor replaced three times this month — is that normal?",
       pattern="count", group="ild_rotor_3plus",
       expect_count_tier="count:3+",
       desc="ILD rotor 3x replacement — cached under count:3+ key"),

    # ── Pattern 6: Compound condition ─────────────────────────────────────────

    TC("Metal temperature is high — what should I do?",
       pattern="compound", group="metal_temp_single",
       expect_compound=False,
       desc="Single fault — cacheable"),

    TC("Metal temp is running above setpoint — steps?",
       pattern="compound", group="metal_temp_single",
       expect_compound=False, expect_hit_after="metal_temp_single",
       desc="Single fault paraphrase — must HIT"),

    TC("Metal temperature is high AND H2 is also elevated — what do I do?",
       pattern="compound", group="compound_temp_h2",
       expect_compound=True,
       desc="Compound fault — gets its own cache entry"),

    TC("H2 is high plus the ILD flow is also low — what now?",
       pattern="compound", group="compound_h2_ild",
       expect_compound=True,
       desc="Compound H2 + ILD — own cache entry"),

    TC("Cooling water is low and at the same time the casting speed is too high",
       pattern="compound", group="compound_cw_speed",
       expect_compound=True,
       desc="Compound cooling + speed — own cache entry"),

    TC("Humidity is high along with low ILD gas flow — both worsening H2",
       pattern="compound", group="compound_humidity_ild",
       expect_compound=True,
       desc="Compound humidity + ILD — own cache entry"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
SEP  = "─" * 72


def _check(tc: TC, result, group_first_keys: dict) -> dict:
    sig    = result.signals
    checks = {}
    fails  = []

    # ── Range tag
    if tc.expect_range_tag:
        ok = tc.expect_range_tag in sig.numeric_ranges
        checks["range_tag"] = PASS if ok else FAIL
        if not ok:
            fails.append(f"Expected range '{tc.expect_range_tag}' but got {sig.numeric_ranges}")

    # ── Named entity
    if tc.expect_entity:
        ok = tc.expect_entity in sig.named_entities
        checks["entity"] = PASS if ok else FAIL
        if not ok:
            fails.append(f"Expected entity '{tc.expect_entity}' but got {sig.named_entities}")

    # ── Negation
    if tc.expect_negation:
        ok = sig.negation_flag == tc.expect_negation
        checks["negation"] = PASS if ok else FAIL
        if not ok:
            fails.append(f"Expected negation='{tc.expect_negation}' but got '{sig.negation_flag}'")

    # ── Count tier
    if tc.expect_count_tier:
        ok = sig.count_tier == tc.expect_count_tier
        checks["count_tier"] = PASS if ok else FAIL
        if not ok:
            fails.append(f"Expected count_tier='{tc.expect_count_tier}' but got '{sig.count_tier}'")

    # ── Compound flag
    if tc.expect_compound:
        ok = sig.compound_flag
        checks["compound"] = PASS if ok else FAIL
        if not ok:
            fails.append("Expected compound=True but got False")

    # ── Cache HIT: same-group paraphrase must share a key
    if tc.expect_hit_after:
        expected_key = group_first_keys.get(tc.expect_hit_after, "")
        if not expected_key:
            checks["cache_hit"] = SKIP
        elif result.cache_hit and sig.composite_key == expected_key:
            checks["cache_hit"] = PASS
        else:
            checks["cache_hit"] = FAIL
            fails.append(
                f"Expected HIT on group '{tc.expect_hit_after}' "
                f"(key={expected_key[:28]}…) but source='{result.source}', "
                f"key='{sig.composite_key[:28]}…'"
            )

    # ── Group isolation: different groups must not share keys
    if tc.group in group_first_keys:
        same_family = {
            g: k for g, k in group_first_keys.items()
            if g != tc.group and g.split("_")[0] == tc.group.split("_")[0]
        }
        for other_group, other_key in same_family.items():
            if sig.composite_key == other_key and not tc.expect_hit_after:
                checks["isolation"] = FAIL
                fails.append(f"Key collision with '{other_group}' — different groups must have different keys")
                break
        else:
            checks["isolation"] = PASS

    overall = FAIL if any(v == FAIL for v in checks.values()) else PASS
    return {
        "desc":    tc.desc, "pattern": tc.pattern, "group": tc.group,
        "query":   tc.query[:68] + "…" if len(tc.query) > 68 else tc.query,
        "source":  result.source,
        "key":     sig.composite_key[:30] + "…",
        "ranges":  sig.numeric_ranges, "entities": sig.named_entities,
        "negation":sig.negation_flag, "count_tier": sig.count_tier,
        "compound":sig.compound_flag, "latency_ms": result.lookup_latency_ms,
        "checks":  checks, "fails": fails, "overall": overall,
    }


def run_tests(filter_pattern: str = "", verbose: bool = False) -> dict:
    interceptor      = PreCacheInterceptor()
    group_first_keys = {}
    results          = []
    pattern_stats    = defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0})
    latencies        = []

    cases = [tc for tc in TEST_CASES
             if not filter_pattern or tc.pattern == filter_pattern]

    print(f"\n{'=' * 72}")
    print(f"  AIM Pre-Cache Interceptor — Test Suite (no-bypass)")
    print(f"  {len(cases)} test cases | patterns: "
          f"{', '.join(sorted(set(tc.pattern for tc in cases)))}")
    print(f"{'=' * 72}\n")

    for tc in cases:
        result = interceptor.process(tc.query)
        latencies.append(result.lookup_latency_ms)

        if tc.group not in group_first_keys:
            group_first_keys[tc.group] = result.signals.composite_key

        record = _check(tc, result, group_first_keys)
        results.append(record)

        for v in record["checks"].values():
            if v == PASS:   pattern_stats[tc.pattern]["pass"] += 1
            elif v == FAIL: pattern_stats[tc.pattern]["fail"] += 1
            else:           pattern_stats[tc.pattern]["skip"] += 1

        icon     = "✓" if record["overall"] == PASS else "✗"
        src_icon = {"cache_hit": "⚡", "miss_then_store": "↓"}.get(result.source, "?")
        print(f"  [{icon}] [{tc.pattern:8s}] {src_icon} {record['query']}")

        if verbose or record["overall"] == FAIL:
            print(f"       group={tc.group} | key={record['key']}")
            if record["ranges"]:   print(f"       ranges={record['ranges']}")
            if record["entities"]: print(f"       entities={record['entities']}")
            if record["negation"]: print(f"       negation={record['negation']}")
            if record["count_tier"]: print(f"       count_tier={record['count_tier']}")
            if record["compound"]: print(f"       compound=True")
            for f in record["fails"]: print(f"       !! {f}")
            print()

    total      = len(results)
    total_pass = sum(1 for r in results if r["overall"] == PASS)
    total_fail = total - total_pass
    accuracy   = round(total_pass / total * 100, 1) if total else 0

    avg_lat  = round(sum(latencies) / len(latencies), 3) if latencies else 0
    p99_lat  = round(sorted(latencies)[int(len(latencies) * 0.99)], 3) if latencies else 0
    max_lat  = round(max(latencies), 3) if latencies else 0

    hits    = sum(1 for r in results if r["source"] == "cache_hit")
    stores  = sum(1 for r in results if r["source"] == "miss_then_store")

    print(f"\n{SEP}")
    print(f"  RESULTS")
    print(f"{SEP}")
    print(f"  Total tests          : {total}")
    print(f"  PASSED               : {total_pass}  ({accuracy}%)")
    print(f"  FAILED               : {total_fail}")
    print()
    print(f"  Cache entries stored : {interceptor.store.size()}")
    print(f"  Cache HITs           : {hits}")
    print(f"  New stores (miss→RAG): {stores}")
    print(f"  Bypasses             : 0  (feature removed)")
    print()
    print(f"  Latency (interceptor only):")
    print(f"    avg : {avg_lat} ms")
    print(f"    p99 : {p99_lat} ms")
    print(f"    max : {max_lat} ms")

    print(f"\n{SEP}")
    print(f"  PER-PATTERN ACCURACY")
    print(f"{SEP}")
    for pat in ["numeric","temporal","entity","negation","count","compound"]:
        if pat not in pattern_stats: continue
        s   = pattern_stats[pat]
        tot = s["pass"] + s["fail"]
        pct = round(s["pass"] / tot * 100, 1) if tot else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {pat:10s} [{bar}] {pct:5.1f}%  ({s['pass']}/{tot} checks)")

    if total_fail:
        print(f"\n{SEP}")
        print(f"  FAILED TESTS")
        print(f"{SEP}")
        for r in results:
            if r["overall"] == FAIL:
                print(f"  ✗ [{r['pattern']:8s}] {r['query']}")
                for f in r["fails"]: print(f"       {f}")

    print(f"\n{SEP}")
    return {
        "total": total, "passed": total_pass, "failed": total_fail,
        "accuracy_pct": accuracy,
        "cache_hits": hits, "new_stores": stores, "bypasses": 0,
        "cache_size": interceptor.store.size(),
        "latency": {"avg_ms": avg_lat, "p99_ms": p99_lat, "max_ms": max_lat},
        "per_pattern": dict(pattern_stats), "test_results": results,
    }


def quick_test(query: str, interceptor: PreCacheInterceptor = None) -> None:
    if interceptor is None:
        interceptor = PreCacheInterceptor()
    r = interceptor.process(query)
    s = r.signals
    print(f"\nQuery   : {query}")
    print(f"Source  : {r.source}")
    print(f"Key     : {s.composite_key}")
    print(f"Ranges  : {s.numeric_ranges}")
    print(f"Entities: {s.named_entities}")
    print(f"Negation: {s.negation_flag}")
    print(f"Count   : {s.count_tier}")
    print(f"Compound: {s.compound_flag}")
    print(f"Temporal: {s.temporal_urgency}")
    print(f"Latency : {r.lookup_latency_ms} ms")
    if r.cache_hit:
        print(f"Answer  : {r.cached_answer[:120]}")
    else:
        print(f"RAG stub: {(r.rag_stub_answer or '')[:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIM Cache Interceptor Test Runner")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pattern", default="")
    parser.add_argument("--query",   default="")
    args = parser.parse_args()

    if args.query:
        quick_test(args.query)
        sys.exit(0)

    summary = run_tests(filter_pattern=args.pattern, verbose=args.verbose)
    sys.exit(0 if summary["failed"] == 0 else 1)
