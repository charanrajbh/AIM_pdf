"""
pre_cache_interceptor.py  — AIM Plant Advisor
==============================================
Standalone pre-cache interceptor with all six similarity-trap extractors.

Fully self-contained — zero imports from the main pipeline.
Uses an in-memory dict as the cache store (drop-in replaceable with Redis).

Six extractors implemented:
  1. NumericRangeExtractor    — value → named zone (normal/caution/critical/danger/dump)
  2. TemporalUrgencyExtractor — tense signals → historical | active_now
  3. NamedEntityExtractor     — equipment model / zone / alloy → exact key component
  4. NegationExtractor        — affirmative vs negated flag
  5. CountOrdinalExtractor    — first/second/3+ escalation tier
  6. CompoundConditionExtractor — AND/OR multi-fault detection

NOTE: Bypass logic has been intentionally removed.
Every query — regardless of severity, urgency, or compound nature — always goes
through the full cache lookup path:
  HIT  → return cached answer
  MISS → call RAG stub → store → return answer

Cache key formula:
  "signals|" + "|".join( sorted(signal_tags) )

Usage:
  interceptor = PreCacheInterceptor()
  result = interceptor.process("H2 level is 0.18 ml/100g — is this OK?")
  print(result.cache_hit, result.source)
"""

import re
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedSignals:
    """All signals extracted from a single query."""
    numeric_ranges:   list[str] = field(default_factory=list)  # e.g. ["h2_level:reject"]
    temporal_urgency: str       = ""                           # "historical" | "active_now" | ""
    named_entities:   list[str] = field(default_factory=list)  # e.g. ["model:AIM-5000"]
    negation_flag:    str       = ""                           # ":affirm" | ":negated"
    count_tier:       str       = ""                           # "count:1" | "count:2" | "count:3+"
    compound_flag:    bool      = False
    rewritten_query:  str       = ""                           # raw numbers replaced with labels
    composite_key:    str       = ""                           # final cache key


@dataclass
class CacheResult:
    """Result returned by interceptor.process()."""
    query:             str
    signals:           ExtractedSignals
    cache_hit:         bool
    cached_answer:     Optional[str] = None
    source:            str           = ""    # "cache_hit" | "miss_then_store"
    lookup_latency_ms: float         = 0.0
    rag_stub_answer:   Optional[str] = None  # what RAG stub returned (on miss)
    stored_as_new:     bool          = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Numeric Range Extractor
# ─────────────────────────────────────────────────────────────────────────────

# Each param: unit_pattern, list of (lo, hi, label)
# never_cache flags removed — all ranges are now cacheable
NUMERIC_THRESHOLDS: dict[str, dict] = {
    "h2_level": {
        "display":       "H2 level (ml/100g Al)",
        "unit_pattern":  r"ml\s*/\s*100\s*g",
        "param_pattern": r"(?:h2|hydrogen)\s*(?:level|content|reading|value|is|at|of)?",
        "ranges": [
            (0,    0.10, "normal"),
            (0.10, 0.15, "caution"),
            (0.15, 0.20, "reject"),
            (0.20, 99,   "dump"),
        ],
    },
    "metal_temp": {
        "display":       "Metal temperature (°C)",
        "unit_pattern":  r"°\s*[cC]|deg(?:rees?)?\s*[cC]|celsius",
        "param_pattern": r"(?:metal|melt|holding\s*furnace|tundish)\s*temp(?:erature)?",
        "ranges": [
            (685,  740,  "normal"),
            (740,  760,  "high"),
            (760,  800,  "trip"),
            (800,  9999, "danger"),
        ],
    },
    "compressor_temp": {
        "display":       "Compressor temperature (°C)",
        "unit_pattern":  r"°\s*[cC]|deg(?:rees?)?\s*[cC]|celsius",
        "param_pattern": r"(?:compressor|motor|pump)\s*temp(?:erature)?",
        "ranges": [
            (0,    50,   "normal"),
            (50,   60,   "caution"),
            (60,   70,   "critical"),
            (70,   9999, "danger"),
        ],
    },
    "cooling_water_flow": {
        "display":       "Cooling water flow (%)",
        "unit_pattern":  r"%\s*(?:of\s*)?(?:target|setpoint|flow)?|percent",
        "param_pattern": r"cooling\s*water\s*flow|water\s*flow|primary\s*cooling",
        "ranges": [
            (80,  100, "normal"),
            (60,  80,  "low"),
            (0,   60,  "trip"),
        ],
    },
    "gas_pressure": {
        "display":       "Gas pressure (mbar)",
        "unit_pattern":  r"mbar|millibar",
        "param_pattern": r"gas\s*pressure|fuel\s*pressure",
        "ranges": [
            (80,  9999, "normal"),
            (60,  80,   "alarm"),
            (0,   60,   "shutdown"),
        ],
    },
    "casting_speed": {
        "display":       "Casting speed (mm/min)",
        "unit_pattern":  r"mm\s*/\s*min",
        "param_pattern": r"casting\s*speed|cast\s*speed|table\s*speed",
        "ranges": [
            (60,  120, "normal"),
            (120, 150, "high"),
            (150, 999, "over_speed"),
        ],
    },
    "ild_gas_flow": {
        "display":       "ILD gas flow (Nl/min)",
        "unit_pattern":  r"nl\s*/\s*min|normal\s*liter",
        "param_pattern": r"(?:ild|degasser?|rotor)\s*(?:gas|n2|nitrogen)?\s*flow",
        "ranges": [
            (8,   20,  "normal"),
            (6,   8,   "low"),
            (0,   6,   "trip"),
        ],
    },
}


def _classify_range(param: str, value: float) -> str:
    """Return range_label for a param+value pair."""
    for lo, hi, label in NUMERIC_THRESHOLDS[param]["ranges"]:
        if lo <= value < hi:
            return label
    return NUMERIC_THRESHOLDS[param]["ranges"][-1][2]


class NumericRangeExtractor:
    """Extracts numeric values, classifies into threshold ranges, rewrites query."""

    def extract(self, query: str) -> tuple[list[str], str]:
        """
        Returns:
            range_tags : list of "param:label" strings
            rewritten  : query with raw numbers replaced by [param:label]
        """
        lower      = query.lower()
        rewritten  = query
        range_tags = []

        for param, cfg in NUMERIC_THRESHOLDS.items():
            if not re.search(cfg["param_pattern"], lower, re.IGNORECASE):
                continue

            unit_pat = cfg["unit_pattern"]
            full_pat = re.compile(
                r"(\d+\.?\d*)\s*(?:" + unit_pat + r")", re.IGNORECASE
            )
            matches = full_pat.findall(query)
            if not matches:
                near_pat = re.compile(
                    cfg["param_pattern"] + r"[^.]{0,40}?(\d+\.?\d*)", re.IGNORECASE
                )
                near = near_pat.search(lower)
                if near and near.group(1):
                    matches = [near.group(1)]

            for raw_val in matches:
                if raw_val is None:
                    continue
                value = float(raw_val)
                label = _classify_range(param, value)
                tag   = f"{param}:{label}"
                if tag not in range_tags:
                    range_tags.append(tag)
                rewritten = re.sub(
                    r"\b" + re.escape(raw_val) + r"\b",
                    f"[{tag}]",
                    rewritten,
                    count=1,
                )

        return range_tags, rewritten


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal Urgency Extractor
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_NOW_PATTERNS = [
    r"\bright\s*now\b",
    r"\bcurrently\b",
    r"\bat\s*this\s*moment\b",
    r"\bjust\s*(tripped|failed|happened|occurred|went)\b",
    r"\bis\s+(tripping|failing|happening|alarming)\b",
    r"\bare\s+(tripping|failing|happening|alarming)\b",
    r"\bgoing\s*on\s*now\b",
    r"\bactive\s+(fault|alarm|trip)\b",
    r"\bstill\s+(tripping|failing|alarming)\b",
]

_HISTORICAL_PATTERNS = [
    r"\blast\s+(week|tuesday|monday|wednesday|thursday|friday|saturday|sunday|month|year|heat|cast|shift)\b",
    r"\byesterday\b",
    r"\bpreviously\b",
    r"\blast\s+time\b",
    r"\b(did|was|were|had|happened|occurred)\b",
    r"\b(earlier|before|ago)\b",
    r"\brecently\b",
    r"\bin\s+the\s+past\b",
]


class TemporalUrgencyExtractor:
    """Classifies query tense — used as a cache key signal, not for bypass."""

    def extract(self, query: str) -> str:
        """Returns urgency_label: 'active_now' | 'historical' | ''"""
        lower = query.lower()
        if any(re.search(p, lower) for p in _ACTIVE_NOW_PATTERNS):
            return "active_now"
        if any(re.search(p, lower) for p in _HISTORICAL_PATTERNS):
            return "historical"
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Named Entity Extractor
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_PATTERNS: list[tuple[str, str, str]] = [
    ("model",   r"\bAIM[-\s]?5000\b",  "model:AIM-5000"),
    ("model",   r"\bAIM[-\s]?7500\b",  "model:AIM-7500"),
    ("model",   r"\bAIM[-\s]?10000\b", "model:AIM-10000"),

    ("zone",    r"\bzone\s*1\b",       "zone:1"),
    ("zone",    r"\bzone\s*2\b",       "zone:2"),
    ("zone",    r"\bzone\s*3\b",       "zone:3"),
    ("zone",    r"\bburner\s*1\b",     "burner:1"),
    ("zone",    r"\bburner\s*2\b",     "burner:2"),
    ("zone",    r"\bburner\s*3\b",     "burner:3"),

    ("alloy",   r"\b1050\b",           "alloy:1050"),
    ("alloy",   r"\b1060\b",           "alloy:1060"),
    ("alloy",   r"\b3003\b",           "alloy:3003"),
    ("alloy",   r"\b5052\b",           "alloy:5052"),
    ("alloy",   r"\b5083\b",           "alloy:5083"),
    ("alloy",   r"\b6060\b",           "alloy:6060"),
    ("alloy",   r"\b6061\b",           "alloy:6061"),
    ("alloy",   r"\b6063\b",           "alloy:6063"),
    ("alloy",   r"\b8011\b",           "alloy:8011"),

    ("fault",   r"\bF-0\d\d\b",        None),
    ("section", r"\b§\s*\d+\.?\d*\b",  None),
]


class NamedEntityExtractor:

    def extract(self, query: str) -> list[str]:
        entities = []
        for category, pattern, label in _ENTITY_PATTERNS:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                if label:
                    entities.append(label)
                else:
                    entities.append(f"{category}:{m.group(0).upper().replace(' ', '')}")
        return entities


# ─────────────────────────────────────────────────────────────────────────────
# 4. Negation Extractor
# ─────────────────────────────────────────────────────────────────────────────

_NEGATION_PATTERNS = [
    r"\b(not|never|no longer|isn'?t|aren'?t|can'?t|don'?t|shouldn'?t|won'?t|doesn'?t)\b",
    r"\b(without|unless|except)\b",
    r"\bnot\s+safe\b",
    r"\bnot\s+acceptable\b",
    r"\bnot\s+recommended\b",
    r"\bforbidden\b",
    r"\bprohibited\b",
    r"\bdo\s+not\b",
    r"\bavoid\b",
    r"\bprevent\b",
    r"\bnever\b",
]


class NegationExtractor:

    def extract(self, query: str) -> str:
        lower = query.lower()
        if any(re.search(p, lower) for p in _NEGATION_PATTERNS):
            return ":negated"
        return ":affirm"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Count / Ordinal Extractor
# ─────────────────────────────────────────────────────────────────────────────

_COUNT_TIERS: list[tuple[str, list[str]]] = [
    ("count:repeated", [r"\brepeat\w*\b", r"\bfrequent\w*\b", r"\bchronic\b",
                        r"\bpattern\b", r"\bconstantly\b",
                        r"\bkeeps?\s+(failing|tripping|happening|occurring)\b",
                        r"\bevery\s+(cast|heat|time|shift)\b",
                        r"\balways\s+(fail|trip|alarm)\b"]),
    ("count:3+",       [r"\bthree\b", r"\b3\s*(times?|attempts?|failures?)\b",
                        r"\bthird\b", r"\b3rd\b", r"\bin\s*a\s*row\b",
                        r"\bmultiple\s+times\b"]),
    ("count:2",        [r"\bsecond\b", r"\btwice\b", r"\b2nd\b", r"\btwo\s*times\b",
                        r"\bagain\b"]),
    ("count:1",        [r"\bfirst\s*time\b", r"\bonce\b", r"\b1st\b", r"\bone\s*time\b",
                        r"\bfor\s*the\s*first\b"]),
]


class CountOrdinalExtractor:

    def extract(self, query: str) -> str:
        """Returns count tier string, or '' if none detected."""
        lower = query.lower()
        for tier, patterns in _COUNT_TIERS:
            if any(re.search(p, lower) for p in patterns):
                return tier
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. Compound Condition Extractor
# ─────────────────────────────────────────────────────────────────────────────

_COMPOUND_MARKERS = [
    r"\band\b", r"\bas\s*well\s*as\b", r"\balso\b", r"\bplus\b", r"\bboth\b",
    r"\bat\s*the\s*same\s*time\b", r"\bsimultaneously\b", r"\bwhile\b",
    r"\balong\s*with\b", r"\bcombined\s*with\b", r"\bon\s*top\s*of\b",
]

_FAULT_NOUNS = [
    "h2", "hydrogen", "temperature", "temp",
    "cooling water", "water flow", "gas pressure",
    "ild", "rotor", "burner", "casting speed",
    "metal level", "mould oil", "dross",
    "humidity", "nitrogen", "n2",
]


class CompoundConditionExtractor:

    def extract(self, query: str, all_numeric_ranges: list[str]) -> bool:
        """Returns True if multiple fault domains detected together."""
        lower = query.lower()
        has_conjunction = any(re.search(p, lower) for p in _COMPOUND_MARKERS)
        if not has_conjunction:
            return False
        unique_nouns  = len(set(n for n in _FAULT_NOUNS if n in lower))
        unique_params = len(set(t.split(":")[0] for t in all_numeric_ranges))
        return max(unique_nouns, unique_params) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Composite Cache Key Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_composite_key(signals: ExtractedSignals) -> str:
    """
    Build key purely from sorted signal tags.

    Two paraphrased queries extracting the same signals → identical key → HIT.
    Different signals (different range zone, different entity, etc.) → different key → MISS.

    Key format: "signals|<tag_1>|<tag_2>|..."
    Example:
        "H2 is 0.08 ml/100g"     → signals|:affirm|h2_level:normal
        "Hydrogen at 0.09 ml/100g" → signals|:affirm|h2_level:normal  ← same → HIT
        "H2 is 0.17 ml/100g"     → signals|:affirm|h2_level:reject    ← different → MISS
    """
    tags = sorted(
        signals.numeric_ranges
        + signals.named_entities
        + ([f"temporal:{signals.temporal_urgency}"] if signals.temporal_urgency else [])
        + ([signals.negation_flag]                  if signals.negation_flag else [])
        + ([signals.count_tier]                     if signals.count_tier else [])
        + (["compound:true"]                        if signals.compound_flag else [])
    )
    # For compound queries with no numeric signals, include a hash of the
    # rewritten query so each distinct compound fault gets its own entry.
    if signals.compound_flag and not signals.numeric_ranges:
        stripped = re.sub(r"\s+", " ", signals.rewritten_query.lower().strip())
        q_hash   = hashlib.sha256(stripped.encode()).hexdigest()[:10]
        tags.append(f"qhash:{q_hash}")
    tag_str = "|".join(tags) if tags else "raw"
    return f"signals|{tag_str}"


# ─────────────────────────────────────────────────────────────────────────────
# TTL table — severity → cache freshness
# ─────────────────────────────────────────────────────────────────────────────

def _get_ttl(signals: ExtractedSignals) -> int:
    """
    Returns TTL in seconds.
    Higher-severity zones get shorter TTL so answers refresh more often.
    """
    high_severity_labels = {"caution", "high", "low", "alarm", "reject",
                            "critical", "danger", "dump", "trip", "shutdown", "over_speed"}
    for tag in signals.numeric_ranges:
        _, label = tag.split(":", 1)
        if label in high_severity_labels:
            return 3600  # 1 hour

    if signals.temporal_urgency == "active_now":
        return 1800  # 30 min — active queries age faster

    if signals.temporal_urgency == "historical":
        return 86400 * 7  # 7 days — historical root-cause answers stay fresh longer

    return 86400  # default 24 hours


# ─────────────────────────────────────────────────────────────────────────────
# RAG Stub — simulates what the pipeline would return (for testing)
# ─────────────────────────────────────────────────────────────────────────────

_RAG_STUB_RESPONSES: dict[str, str] = {
    "h2_level:normal":           "TYPE: FACT\nH2 ≤0.10 ml/100g Al — within spec. Continue casting. Monitor with next Alspek reading.",
    "h2_level:caution":          "TYPE: FAULT\nH2 elevated (0.10–0.15). Extend ILD pass by 5 min, increase N2 to 12 Nl/min. Re-sample before casting.",
    "h2_level:reject":           "TYPE: FAULT\nH2 above reject limit (>0.15). STOP cast. Complete second ILD pass at 15 Nl/min. Do not cast >0.20.",
    "h2_level:dump":             "TYPE: FAULT\nH2 emergency dump zone (>0.20). Divert melt. Do not cast under any circumstance.",
    "compressor_temp:normal":    "TYPE: FACT\nCompressor temp ≤50°C — normal. No action needed. Check air filter monthly.",
    "compressor_temp:caution":   "TYPE: FAULT\nCompressor temp 50–60°C. Clean intake filter. Ensure 300mm ventilation clearance.",
    "compressor_temp:critical":  "TYPE: FAULT\nCompressor temp 60–70°C — CRITICAL. STOP compressor. 30 min cool-down. Inspect motor windings.",
    "compressor_temp:danger":    "TYPE: FAULT\nCompressor temp >70°C — DANGER. Motor burnout risk. Shut down immediately. Do not restart until inspected.",
    "metal_temp:normal":         "TYPE: FACT\nMetal temperature 685–740°C — within casting range. See Table 3-B for alloy-specific targets.",
    "metal_temp:high":           "TYPE: FAULT\nMetal temp >740°C. Reduce holding furnace setpoint 15–20°C. Equilibrate 20 min. Verify with thermocouple. See F-012.",
    "metal_temp:trip":           "TYPE: FAULT\nMetal temp >760°C — F-012 DCS trip zone. Reduce setpoint immediately. Notify shift supervisor.",
    "metal_temp:danger":         "TYPE: FAULT\nMetal temp >800°C — DANGER. Thermal runaway risk. Emergency action required.",
    "cooling_water_flow:normal": "TYPE: FACT\nCooling water flow ≥80% setpoint — normal. Continue operation.",
    "cooling_water_flow:low":    "TYPE: FAULT\nCooling water flow 60–80%. Check pump pressure and suction strainer. F-021 approaching.",
    "cooling_water_flow:trip":   "TYPE: FAULT\nCooling water flow <60% — F-021 trip zone. Check distribution manifold for blockage.",
    "ild_gas_flow:normal":       "TYPE: FACT\nILD N2 flow 8–20 Nl/min — within spec. Continue degassing.",
    "ild_gas_flow:low":          "TYPE: FAULT\nILD N2 flow 6–8 Nl/min — F-011 approaching. Check N2 supply pressure and rotor condition.",
    "ild_gas_flow:trip":         "TYPE: FAULT\nILD N2 flow <6 Nl/min — F-011 trip zone. Stop degassing. Inspect rotor and gas line.",
}


def _rag_stub(signals: ExtractedSignals, original_query: str) -> str:
    """Return best-matching stub response for the extracted signals."""
    for tag in signals.numeric_ranges:
        if tag in _RAG_STUB_RESPONSES:
            return _RAG_STUB_RESPONSES[tag]
    return f"[RAG STUB] Routed to RAG pipeline for: '{original_query[:80]}'"


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Cache Store  (swap for Redis in production)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryCacheStore:
    """
    In-memory key-value cache with TTL.
    Replace get/set with Redis calls in production:
        redis.get(key) / redis.setex(key, ttl, value)
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    def get(self, key: str) -> Optional[dict]:
        entry = self._store.get(key)
        if entry and (entry["ttl"] == 0 or time.time() < entry["expires_at"]):
            return entry
        return None

    def set(self, key: str, value: str, query: str, signals: ExtractedSignals,
            ttl_seconds: int = 86400) -> None:
        self._store[key] = {
            "key":        key,
            "value":      value,
            "query":      query,
            "signals":    asdict(signals),
            "stored_at":  time.time(),
            "expires_at": time.time() + ttl_seconds if ttl_seconds else 0,
            "ttl":        ttl_seconds,
            "hit_count":  0,
        }

    def increment_hit(self, key: str) -> None:
        if key in self._store:
            self._store[key]["hit_count"] += 1

    def all_entries(self) -> list[dict]:
        return list(self._store.values())

    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Cache Interceptor  — main entry point
# ─────────────────────────────────────────────────────────────────────────────

class PreCacheInterceptor:
    """
    Runs all six extractors → builds composite key → checks/stores in cache.

    Every query always hits the cache lookup. No bypass path exists.

    Flow:
        query → extractors → composite key
                                  ↓
                        HIT  → return cached answer
                        MISS → call RAG → store → return answer

    Usage:
        interceptor = PreCacheInterceptor()
        result = interceptor.process("H2 level is 0.18 ml/100g — is this OK?")
        print(result.cache_hit, result.source)
    """

    def __init__(self, cache_store: Optional[InMemoryCacheStore] = None):
        self.store    = cache_store or InMemoryCacheStore()
        self.numeric  = NumericRangeExtractor()
        self.temporal = TemporalUrgencyExtractor()
        self.entity   = NamedEntityExtractor()
        self.negation = NegationExtractor()
        self.count    = CountOrdinalExtractor()
        self.compound = CompoundConditionExtractor()

    def _run_extractors(self, query: str) -> ExtractedSignals:
        sig = ExtractedSignals()

        # 1. Numeric — also rewrites query (raw numbers → [param:label])
        sig.numeric_ranges, sig.rewritten_query = self.numeric.extract(query)

        # 2. Temporal urgency — used as cache key signal only
        sig.temporal_urgency = self.temporal.extract(query)

        # 3. Named entities
        sig.named_entities = self.entity.extract(query)

        # 4. Negation flag
        sig.negation_flag = self.negation.extract(query)

        # 5. Count / ordinal tier
        sig.count_tier = self.count.extract(query)

        # 6. Compound flag (informational — included in key so compound queries
        #    get their own cache entry separate from single-fault entries)
        sig.compound_flag = self.compound.extract(query, sig.numeric_ranges)

        # Build composite key from all signals
        sig.composite_key = _build_composite_key(sig)

        return sig

    def process(self, query: str) -> CacheResult:
        t0  = time.perf_counter()
        sig = self._run_extractors(query)

        result = CacheResult(query=query, signals=sig, cache_hit=False)

        # Cache lookup — always attempted, no bypass
        entry = self.store.get(sig.composite_key)
        if entry:
            self.store.increment_hit(sig.composite_key)
            result.cache_hit         = True
            result.cached_answer     = entry["value"]
            result.source            = "cache_hit"
            result.lookup_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return result

        # Cache miss — call RAG stub, store result
        stub = _rag_stub(sig, query)
        result.rag_stub_answer = stub
        result.source          = "miss_then_store"
        result.stored_as_new   = True

        ttl = _get_ttl(sig)
        self.store.set(sig.composite_key, stub, query, sig, ttl_seconds=ttl)

        result.lookup_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return result

    def get_cache_contents(self) -> list[dict]:
        return self.store.all_entries()

    def clear_cache(self) -> None:
        self.store.clear()
