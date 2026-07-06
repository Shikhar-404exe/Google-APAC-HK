"""
PROJECT SKYPAD - STAGE 2: AI REASONING AND SPATIAL ROUTING LAYER
routing.py

Gemini 1.5 Flash (Vertex AI) intent-extraction and coordinate-mapping layer.
Converts raw community crisis text into a structured routing directive that
the Stage 1 physics engine (engine.py) can act on.

Data-flow role:
    [Raw community text]
         |
         v  (Vertex AI - Gemini 1.5 Flash)
    [CrisisIntent: crisis type, severity, location cues]
         |
         v  (Spatial Mapper)
    [SpatialRoute: normalised X, Y, Z within AOS simulation grid]
         |
         v  (FastAPI /route endpoint -> engine.py /stabilize)
    [AOSResponse: acoustic vector commands]

Simulation grid definition:
    X, Y : 0.0 - 100.0 m  (city-block-scale urban patch)
    Z    : 3.5 - 12.0 m   (AOS operational altitude corridor from AOS_RULES.md)

Usage:
    python routing.py                         # demo mode (no Vertex AI needed)
    GOOGLE_CLOUD_PROJECT=my-project-id \\
    python routing.py                         # live Vertex AI mode
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skypad.routing")

# ---------------------------------------------------------------------------
# 1.  CONFIGURATION
# ---------------------------------------------------------------------------

# Google Cloud credentials – set via environment variables or a .env file.
# The module fails fast on startup if these are wrong, avoiding silent errors.
GCP_PROJECT:  str = os.getenv("GOOGLE_CLOUD_PROJECT",  "your-gcp-project-id")
GCP_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Simulation grid spatial bounds (must mirror AOS_RULES.md and engine.py)
GRID_X_MIN, GRID_X_MAX = 0.0,  100.0   # metres West-East
GRID_Y_MIN, GRID_Y_MAX = 0.0,  100.0   # metres South-North
GRID_Z_MIN, GRID_Z_MAX = 3.5,  12.0    # metres altitude (AOS corridor)

# Gemini generation settings: low temperature = deterministic, factual output
GENERATION_CONFIG: dict = {
    "temperature":       0.1,
    "top_p":             0.9,
    "max_output_tokens": 1024,
}

# ---------------------------------------------------------------------------
# 2.  DATA MODELS
# ---------------------------------------------------------------------------

class CrisisType(str, Enum):
    """Taxonomy of crisis categories the AOS can respond to."""
    MEDICAL         = "MEDICAL"
    HEATWAVE        = "HEATWAVE"
    WATER_SHORTAGE  = "WATER_SHORTAGE"
    POWER_OUTAGE    = "POWER_OUTAGE"
    CROWD_CRUSH     = "CROWD_CRUSH"
    AIR_QUALITY     = "AIR_QUALITY"
    FLOOD           = "FLOOD"
    UNKNOWN         = "UNKNOWN"


class SeverityLevel(str, Enum):
    """Standardised severity levels (mirrors hospital triage colours)."""
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class CrisisIntent(BaseModel):
    """
    Structured intent extracted from raw community text by Gemini 1.5 Flash.
    Pure AI output layer – no physics computations here.
    """
    crisis_type:       CrisisType
    severity:          SeverityLevel
    severity_score:    float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalised severity scalar [0.0 = benign .. 1.0 = critical]"
    )
    location_cues:     list[str]     = Field(
        default_factory=list,
        description="Raw location tokens extracted from text (streets, landmarks)"
    )
    affected_count:    Optional[int] = Field(
        None,
        description="Estimated number of people affected (None if unclear)"
    )
    urgency_keywords:  list[str]     = Field(
        default_factory=list,
        description="Verbatim urgency words found in the crisis text"
    )
    required_asset:    str = Field(
        ...,
        description="Recommended utility block type dispatched to the scene"
    )
    reasoning:         str = Field(
        ...,
        description="One-sentence rationale for the classification"
    )


class NormalisedCoordinate(BaseModel):
    """
    AOS grid coordinate X, Y, Z.
    Z is always within the operational corridor [GRID_Z_MIN, GRID_Z_MAX].
    """
    x: float = Field(..., ge=GRID_X_MIN, le=GRID_X_MAX)
    y: float = Field(..., ge=GRID_Y_MIN, le=GRID_Y_MAX)
    z: float = Field(..., ge=GRID_Z_MIN, le=GRID_Z_MAX)


class SpatialRoute(BaseModel):
    """
    Complete routing directive emitted by Stage 2.
    This is the contract consumed by Stage 1 (engine.py / POST /stabilize).

    Fields:
        crisis_intent        – AI-extracted crisis classification
        dispatch_coordinate  – Grid position to send the utility block
        staging_coordinate   – Grid position where the block idles between missions
        priority_rank        – 1 = most critical, 10 = lowest priority
        estimated_transit_s  – Seconds to reach dispatch from staging (at V_max)
        raw_input_text       – Original community report (for audit)
        model_used           – Gemini model identifier
        timestamp_utc        – Unix epoch timestamp
        routing_mode         – 'live' (Vertex AI) or 'demo' (rule-based fallback)
    """
    crisis_intent:        CrisisIntent
    dispatch_coordinate:  NormalisedCoordinate
    staging_coordinate:   NormalisedCoordinate
    priority_rank:        int   = Field(..., ge=1)
    estimated_transit_s:  float = Field(..., ge=0.0)
    raw_input_text:       str
    model_used:           str
    timestamp_utc:        float
    routing_mode:         str


# ---------------------------------------------------------------------------
# 3.  AOS SYSTEM PROMPT  (the Gemini brain for Stage 2)
# ---------------------------------------------------------------------------

AOS_SYSTEM_PROMPT = (
    "You are the Acoustic Operating System (AOS) Spatial Intelligence Core "
    "for Project Skypad - a crisis-response platform that dispatches "
    "levitating utility blocks (medical kits, shade sails, water filtration "
    "units) to urban emergencies.\n\n"
    "YOUR ROLE:\n"
    "Parse raw community crisis reports and output a STRICT JSON object. "
    "No prose, no explanation, no markdown - only the raw JSON object.\n\n"
    "SIMULATION GRID BOUNDS:\n"
    "  X: 0.0 - 100.0 metres  (West-East axis of the monitored urban patch)\n"
    "  Y: 0.0 - 100.0 metres  (South-North axis)\n"
    "  Z: 3.5 - 12.0 metres   (AOS operational altitude corridor, NEVER below 3.5 m)\n\n"
    "SPATIAL MAPPING RULES:\n"
    "1. Parse location cues (street names, cardinal directions, landmarks)\n"
    "   and map them to plausible (X, Y) grid coordinates within [0, 100] m.\n"
    "2. Choose Z based on severity:\n"
    "     LOW      -> Z = 4.5 m  (low hover; minimal disturbance)\n"
    "     MODERATE -> Z = 6.5 m  (standard dispatch altitude)\n"
    "     HIGH     -> Z = 9.0 m  (elevated for wide coverage)\n"
    "     CRITICAL -> Z = 11.5 m (max altitude for maximum broadcast range)\n"
    "3. If no precise location is given, use the grid centroid: X=50, Y=50.\n"
    "4. staging_coordinate is the block idle point: X=5, Y=5, Z=3.5.\n\n"
    "ASSET SELECTION RULES:\n"
    "  MEDICAL        -> medical_kit\n"
    "  HEATWAVE       -> shade_sail\n"
    "  WATER_SHORTAGE -> water_unit\n"
    "  POWER_OUTAGE   -> power_cell\n"
    "  CROWD_CRUSH    -> shade_sail  (crowd dispersal and cooling)\n"
    "  AIR_QUALITY    -> air_filter_unit\n"
    "  FLOOD          -> emergency_beacon\n"
    "  UNKNOWN        -> medical_kit  (safe default)\n\n"
    "SEVERITY SCORING (severity_score, float 0.0 to 1.0):\n"
    "  LOW=0.10-0.30 | MODERATE=0.31-0.60 | HIGH=0.61-0.85 | CRITICAL=0.86-1.00\n\n"
    "OUTPUT FORMAT (strict JSON only, no surrounding text):\n"
    "{\n"
    '  "crisis_type":      "<CrisisType enum value>",\n'
    '  "severity":         "<SeverityLevel enum value>",\n'
    '  "severity_score":   <float 0.0-1.0>,\n'
    '  "location_cues":    ["<token>", ...],\n'
    '  "affected_count":   <int or null>,\n'
    '  "urgency_keywords": ["<word>", ...],\n'
    '  "required_asset":   "<asset string>",\n'
    '  "reasoning":        "<one sentence>",\n'
    '  "dispatch_x":       <float>,\n'
    '  "dispatch_y":       <float>,\n'
    '  "dispatch_z":       <float>\n'
    "}"
)

# ---------------------------------------------------------------------------
# 4.  VERTEX AI CLIENT  (lazy-initialised to avoid import-time crashes)
# ---------------------------------------------------------------------------

_vertex_model = None   # Cached GenerativeModel instance


def _get_vertex_model():
    """
    Initialise and cache the Vertex AI GenerativeModel.
    The AOS system prompt is bound as a system_instruction so every
    generate_content() call automatically carries the AOS context.

    Raises:
        ImportError: if google-cloud-aiplatform is not installed.
        google.api_core.exceptions.*: on authentication or quota errors.
    """
    global _vertex_model
    if _vertex_model is not None:
        return _vertex_model

    import vertexai                                           # type: ignore
    from vertexai.generative_models import GenerativeModel   # type: ignore

    log.info(
        "Initialising Vertex AI  project=%s  location=%s  model=%s",
        GCP_PROJECT, GCP_LOCATION, GEMINI_MODEL,
    )
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    _vertex_model = GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=AOS_SYSTEM_PROMPT,
    )
    return _vertex_model


# ---------------------------------------------------------------------------
# 5.  SPATIAL COORDINATE MAPPER
# ---------------------------------------------------------------------------

# Keyword -> (X, Y) lookup table.
# In production this would query a geocoding API; here we use a deterministic
# table so demo runs are reproducible without any network access.
_LOCATION_GRID_MAP: dict[str, tuple[float, float]] = {
    # Cardinal / relative directions
    "north":     (50.0, 85.0),  "south":     (50.0, 15.0),
    "east":      (85.0, 50.0),  "west":      (15.0, 50.0),
    "northeast": (80.0, 80.0),  "northwest": (20.0, 80.0),
    "southeast": (80.0, 20.0),  "southwest": (20.0, 20.0),
    "central":   (50.0, 50.0),  "centre":    (50.0, 50.0),
    "downtown":  (50.0, 50.0),
    # Street ordinals -> X position (Y fixed at mid-grid)
    "1st": (10.0, 50.0), "2nd": (20.0, 50.0), "3rd": (30.0, 50.0),
    "4th": (40.0, 50.0), "5th": (50.0, 50.0), "6th": (60.0, 50.0),
    "7th": (70.0, 50.0), "8th": (80.0, 50.0), "9th": (90.0, 50.0),
    # Common urban landmarks
    "park":     (35.0, 65.0), "market":   (45.0, 35.0),
    "hospital": (70.0, 70.0), "clinic":   (40.0, 55.0),
    "school":   (25.0, 75.0), "station":  (55.0, 45.0),
    "shelter":  (30.0, 30.0), "mosque":   (60.0, 60.0),
    "temple":   (65.0, 40.0), "church":   (45.0, 70.0),
}

# Severity level -> Z altitude mapping (within the AOS operational corridor)
_SEVERITY_Z_MAP: dict[str, float] = {
    "LOW":      4.5,
    "MODERATE": 6.5,
    "HIGH":     9.0,
    "CRITICAL": 11.5,
}


def _resolve_coordinate(
    location_cues: list[str],
    severity: str,
) -> NormalisedCoordinate:
    """
    Convert AI-extracted location tokens and severity into a grid coordinate.

    Algorithm:
        1. Iterate location_cues; find first token matching _LOCATION_GRID_MAP
           (exact match first, then substring match, case-insensitive).
        2. Default to grid centroid (50, 50) if nothing matches.
        3. Set Z from _SEVERITY_Z_MAP based on severity level.

    Args:
        location_cues: List of raw location tokens from the AI.
        severity:      SeverityLevel string (e.g. "HIGH").

    Returns:
        NormalisedCoordinate within simulation bounds.
    """
    x, y = 50.0, 50.0   # grid centroid default

    for cue in location_cues:
        key = cue.strip().lower()
        if key in _LOCATION_GRID_MAP:
            x, y = _LOCATION_GRID_MAP[key]
            log.debug("Location cue '%s' mapped to (%.1f, %.1f)", cue, x, y)
            break
        # Substring match for compound tokens like "4th street"
        for token, coords in _LOCATION_GRID_MAP.items():
            if token in key:
                x, y = coords
                log.debug("Substring hit '%s' in '%s' -> (%.1f, %.1f)", token, cue, x, y)
                break
        else:
            continue
        break

    z = _SEVERITY_Z_MAP.get(severity, 6.5)
    return NormalisedCoordinate(
        x=float(max(GRID_X_MIN, min(GRID_X_MAX, x))),
        y=float(max(GRID_Y_MIN, min(GRID_Y_MAX, y))),
        z=float(max(GRID_Z_MIN, min(GRID_Z_MAX, z))),
    )


def _estimate_transit_time(
    staging: NormalisedCoordinate,
    dispatch: NormalisedCoordinate,
    v_max_ms: float = 2.0,
) -> float:
    """
    Estimate transit time in seconds from staging to dispatch coordinate.

    Uses Stage 1 V_max = 2.0 m/s straight-line flight:
        t = Euclidean_distance / V_max

    Args:
        staging:   Block idle/home position.
        dispatch:  Crisis target position.
        v_max_ms:  Maximum block speed (m/s); default = Stage 1 V_MAX_MS.

    Returns:
        Estimated transit time in seconds (rounded to 2 dp).
    """
    dist = math.sqrt(
        (dispatch.x - staging.x) ** 2 +
        (dispatch.y - staging.y) ** 2 +
        (dispatch.z - staging.z) ** 2
    )
    return round(dist / v_max_ms, 2)


# ---------------------------------------------------------------------------
# 6.  GEMINI RESPONSE PARSER
# ---------------------------------------------------------------------------

def _parse_gemini_json(raw_text: str) -> dict:
    """
    Robustly extract the JSON object from a Gemini response string.

    Gemini may occasionally wrap output in markdown fences despite instructions.
    This parser strips the fence if present and parses the JSON payload.

    Args:
        raw_text: Raw string returned by the Gemini model.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    text = raw_text.strip()

    # Strip markdown code fence if Gemini disobeyed the no-prose instruction
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the outermost JSON object
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        text = brace_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot parse Gemini JSON.\nRaw: {raw_text!r}\nError: {exc}"
        ) from exc


def _dict_to_crisis_intent(data: dict) -> CrisisIntent:
    """
    Convert a parsed Gemini JSON dict into a validated CrisisIntent model.

    Handles:
        - Graceful fallback for out-of-enum values (unknown crisis type / severity)
        - Coordinate keys (dispatch_x/y/z) that appear in the dict but not in
          CrisisIntent are simply ignored by Pydantic
    """
    valid_crises   = {e.value for e in CrisisType}
    valid_severity = {e.value for e in SeverityLevel}

    crisis_type_raw = str(data.get("crisis_type", "UNKNOWN")).upper()
    severity_raw    = str(data.get("severity",    "MODERATE")).upper()

    crisis_type = crisis_type_raw if crisis_type_raw in valid_crises   else "UNKNOWN"
    severity    = severity_raw    if severity_raw    in valid_severity  else "MODERATE"

    return CrisisIntent(
        crisis_type      = CrisisType(crisis_type),
        severity         = SeverityLevel(severity),
        severity_score   = float(max(0.0, min(1.0, data.get("severity_score", 0.5)))),
        location_cues    = list(data.get("location_cues",   [])),
        affected_count   = data.get("affected_count"),
        urgency_keywords = list(data.get("urgency_keywords", [])),
        required_asset   = str(data.get("required_asset", "medical_kit")),
        reasoning        = str(data.get("reasoning", "No reasoning provided.")),
    )


# ---------------------------------------------------------------------------
# 7.  DEMO / OFFLINE RULE-BASED FALLBACK
# ---------------------------------------------------------------------------

# Keyword sets for rule-based crisis type detection
_CRISIS_KEYWORDS: dict[str, list[str]] = {
    "MEDICAL":        ["faint", "fainting", "injury", "injuries", "ambulance",
                       "clinic", "hospital", "bleeding", "unconscious", "collapse"],
    "HEATWAVE":       ["heat", "sun", "hot", "heatwave", "scorching", "shade",
                       "temperature", "sweat", "dehydrat"],
    "WATER_SHORTAGE": ["water", "thirst", "thirsty", "dehydrat", "drought", "dry"],
    "POWER_OUTAGE":   ["power", "electricity", "blackout", "outage", "dark",
                       "no lights", "generator"],
    "CROWD_CRUSH":    ["crowd", "crush", "stampede", "push", "trampl", "mob",
                       "overcrowded", "packed"],
    "AIR_QUALITY":    ["smoke", "fumes", "toxic", "gas", "aqi", "pollution"],
    "FLOOD":          ["flood", "water level", "rising water", "submerged"],
}

# Keyword sets for severity detection (evaluated in priority order)
_SEVERITY_KEYWORDS: dict[str, list[str]] = {
    "CRITICAL": ["dying", "dead", "mass casualty", "emergency", "immediately",
                 "critical", "life-threatening"],
    "HIGH":     ["faint", "fainting", "collapse", "unconscious", "urgent",
                 "severe", "serious", "hospitalise", "hospitaliz"],
    "MODERATE": ["overcrowded", "long queue", "struggling", "distress",
                 "concerning", "many people"],
    "LOW":      ["minor", "slight", "few", "manageable", "small"],
}

# Crisis type -> recommended utility block asset
_ASSET_MAP: dict[str, str] = {
    "MEDICAL":        "medical_kit",
    "HEATWAVE":       "shade_sail",
    "WATER_SHORTAGE": "water_unit",
    "POWER_OUTAGE":   "power_cell",
    "CROWD_CRUSH":    "shade_sail",
    "AIR_QUALITY":    "air_filter_unit",
    "FLOOD":          "emergency_beacon",
    "UNKNOWN":        "medical_kit",
}


def _demo_rule_based_extraction(text: str) -> dict:
    """
    Offline heuristic extraction when Vertex AI is unavailable.

    Uses keyword frequency scoring for crisis type and priority-order keyword
    matching for severity. Returns the same dict schema Gemini would produce,
    so the rest of the pipeline works identically in both modes.

    Args:
        text: Raw community crisis report.

    Returns:
        dict with the same keys as the Gemini JSON output schema.
    """
    text_lower = text.lower()

    # Score each crisis category by keyword hit count
    scores: dict[str, int] = {k: 0 for k in _CRISIS_KEYWORDS}
    for crisis, kws in _CRISIS_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                scores[crisis] += 1

    crisis_type = max(scores, key=lambda k: scores[k])
    if scores[crisis_type] == 0:
        crisis_type = "UNKNOWN"

    # Severity: walk priority order and break on first keyword match
    severity = "MODERATE"
    for level in ("CRITICAL", "HIGH", "MODERATE", "LOW"):
        for kw in _SEVERITY_KEYWORDS[level]:
            if kw in text_lower:
                severity = level
                break
        if severity != "MODERATE" or level == "LOW":
            break

    score_map = {"LOW": 0.2, "MODERATE": 0.5, "HIGH": 0.75, "CRITICAL": 0.95}
    severity_score = score_map[severity]

    # Extract spatial location tokens via regex
    loc_re = re.compile(
        r"\b(\d+(?:st|nd|rd|th)|north|south|east|west|central|downtown|"
        r"park|market|hospital|clinic|school|station|street|avenue|road|"
        r"lane|square|block|mosque|temple|church|shelter)\b",
        re.IGNORECASE,
    )
    # Preserve insertion order while de-duplicating
    location_cues = list(dict.fromkeys(
        m.group(0).lower() for m in loc_re.finditer(text)
    ))

    # Extract urgency keywords
    urg_re = re.compile(
        r"\b(urgent|immediately|emergency|critical|dying|collapse|fainting|"
        r"serious|severe|danger|help|rescue|overcrowded)\b",
        re.IGNORECASE,
    )
    urgency_keywords = list({m.group(0).lower() for m in urg_re.finditer(text)})

    # Affected count: look for "N people / persons / patients"
    count_m = re.search(
        r"\b(\d+)\s+(?:people|persons|individuals|patients)\b",
        text, re.IGNORECASE,
    )
    affected_count = int(count_m.group(1)) if count_m else None

    # Resolve grid coordinates from location cues
    x, y = 50.0, 50.0
    for cue in location_cues:
        if cue in _LOCATION_GRID_MAP:
            x, y = _LOCATION_GRID_MAP[cue]
            break
        for token, coords in _LOCATION_GRID_MAP.items():
            if token in cue:
                x, y = coords
                break

    z = _SEVERITY_Z_MAP.get(severity, 6.5)

    return {
        "crisis_type":      crisis_type,
        "severity":         severity,
        "severity_score":   severity_score,
        "location_cues":    location_cues,
        "affected_count":   affected_count,
        "urgency_keywords": urgency_keywords,
        "required_asset":   _ASSET_MAP.get(crisis_type, "medical_kit"),
        "reasoning": (
            f"Rule-based demo: detected '{crisis_type}' crisis "
            f"with {severity} severity from keyword analysis."
        ),
        "dispatch_x": x,
        "dispatch_y": y,
        "dispatch_z": z,
    }


# ---------------------------------------------------------------------------
# 8.  CORE ROUTING FUNCTION
# ---------------------------------------------------------------------------

def extract_crisis_and_route(community_text: str) -> SpatialRoute:
    """
    Primary Stage 2 function: raw community text -> SpatialRoute.

    Physics pipeline:
        1. Send community_text to Gemini 1.5 Flash with the AOS system prompt
        2. Parse the strict JSON response into CrisisIntent
        3. Resolve dispatch (X, Y, Z) from location cues and severity level
        4. Compute transit time from staging pad (V_max = 2.0 m/s)
        5. Return a complete SpatialRoute for the Stage 1 physics engine

    Automatic DEMO mode fallback when:
        - GCP_PROJECT is still the placeholder value "your-gcp-project-id"
        - The google-cloud-aiplatform package is not installed
        - Any Vertex AI API error occurs (auth, network, quota)

    Args:
        community_text: Raw, unstructured distress report from a community member.

    Returns:
        SpatialRoute: Full routing directive for the AOS hardware layer.
    """
    log.info("extract_crisis_and_route() called  len=%d", len(community_text))

    gemini_json: dict = {}
    routing_mode = "live"

    # Attempt live Vertex AI call
    if GCP_PROJECT != "your-gcp-project-id":
        try:
            from vertexai.generative_models import GenerationConfig   # type: ignore
            model    = _get_vertex_model()
            gen_cfg  = GenerationConfig(**GENERATION_CONFIG)
            log.info("Calling Gemini 1.5 Flash  model=%s", GEMINI_MODEL)
            response     = model.generate_content(
                contents=community_text, generation_config=gen_cfg
            )
            log.debug("Gemini raw response:\n%s", response.text)
            gemini_json  = _parse_gemini_json(response.text)
            routing_mode = "live"
        except Exception as exc:
            log.warning("Vertex AI failed (%s). Falling back to DEMO mode.", exc)
            routing_mode = "demo"
    else:
        log.info("GCP_PROJECT not configured - running in DEMO mode (rule-based).")
        routing_mode = "demo"

    # Apply demo fallback if live mode failed or was not attempted
    if routing_mode == "demo" or not gemini_json:
        gemini_json  = _demo_rule_based_extraction(community_text)
        routing_mode = "demo"

    # Build validated CrisisIntent from the parsed dict
    crisis_intent = _dict_to_crisis_intent(gemini_json)

    # Resolve dispatch spatial coordinates
    if "dispatch_x" in gemini_json and "dispatch_y" in gemini_json:
        # Use Gemini-provided coordinates (preferred in live mode)
        sev_z = _SEVERITY_Z_MAP.get(crisis_intent.severity.value, 6.5)
        dispatch_coord = NormalisedCoordinate(
            x=max(GRID_X_MIN, min(GRID_X_MAX, float(gemini_json["dispatch_x"]))),
            y=max(GRID_Y_MIN, min(GRID_Y_MAX, float(gemini_json["dispatch_y"]))),
            z=max(GRID_Z_MIN, min(GRID_Z_MAX, float(gemini_json.get("dispatch_z", sev_z)))),
        )
    else:
        # Fallback to the keyword-based spatial resolver
        dispatch_coord = _resolve_coordinate(
            crisis_intent.location_cues, crisis_intent.severity.value
        )

    # Fixed idle / staging pad: north-west corner of the grid
    staging_coord = NormalisedCoordinate(x=5.0, y=5.0, z=3.5)

    # Priority rank: 1 = highest urgency (inverse of severity_score * 10)
    priority_rank = max(1, min(10, round((1.0 - crisis_intent.severity_score) * 10)))

    transit_s = _estimate_transit_time(staging_coord, dispatch_coord)

    route = SpatialRoute(
        crisis_intent       = crisis_intent,
        dispatch_coordinate = dispatch_coord,
        staging_coordinate  = staging_coord,
        priority_rank       = priority_rank,
        estimated_transit_s = transit_s,
        raw_input_text      = community_text,
        model_used          = GEMINI_MODEL,
        timestamp_utc       = time.time(),
        routing_mode        = routing_mode,
    )

    log.info(
        "Route resolved  crisis=%s  severity=%s  score=%.2f  "
        "dispatch=(%.1f,%.1f,%.1f)  transit=%.1fs  mode=%s",
        route.crisis_intent.crisis_type.value,
        route.crisis_intent.severity.value,
        route.crisis_intent.severity_score,
        route.dispatch_coordinate.x,
        route.dispatch_coordinate.y,
        route.dispatch_coordinate.z,
        route.estimated_transit_s,
        route.routing_mode,
    )
    return route


# ---------------------------------------------------------------------------
# 9.  FASTAPI APPLICATION
# ---------------------------------------------------------------------------

router_app = FastAPI(
    title       = "Project Skypad - AOS Routing Layer",
    description = (
        "Stage 2 AI Reasoning and Spatial Routing. "
        "Converts raw community crisis text into structured SpatialRoute "
        "directives via Gemini 1.5 Flash (Vertex AI). "
        "Outputs normalised (X, Y, Z) coordinates for the Stage 1 physics engine."
    ),
    version     = "2.0.0",
)


class RouteRequest(BaseModel):
    """Inbound request payload for POST /route."""
    community_text: str = Field(
        ...,
        min_length=10,
        description="Raw, unstructured community crisis report text.",
        examples=[
            "The medical clinic on 4th street is overcrowded and people are "
            "fainting outside in the sun."
        ],
    )


@router_app.get("/health", tags=["System"])
def routing_health() -> dict:
    """Liveness probe for the routing service."""
    is_demo = GCP_PROJECT == "your-gcp-project-id"
    return {
        "status":       "online",
        "service":      "AOS Routing Layer v2.0.0",
        "stage":        2,
        "model":        GEMINI_MODEL,
        "gcp_project":  GCP_PROJECT,
        "routing_mode": "demo" if is_demo else "live",
        "timestamp_utc": time.time(),
    }


@router_app.post("/route", response_model=SpatialRoute, tags=["AOS Routing"])
def route_crisis(request: RouteRequest) -> SpatialRoute:
    """
    Main routing endpoint.

    POST raw community crisis text; receive a structured SpatialRoute
    with crisis classification, severity score, required asset, and
    dispatch (X, Y, Z) coordinates ready to feed into Stage 1 POST /stabilize.

    Example request body:
        {
            "community_text": "The medical clinic on 4th street is overcrowded
                               and people are fainting outside in the sun."
        }
    """
    try:
        return extract_crisis_and_route(request.community_text)
    except Exception as exc:
        log.error("Routing error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 10. DEMO SCENARIOS  +  LOCAL DEV ENTRYPOINT
# ---------------------------------------------------------------------------

DEMO_SCENARIOS: list[str] = [
    # Scenario 1 – the exact example from the project specification
    "The medical clinic on 4th street is overcrowded and people are fainting outside in the sun.",
    # Scenario 2 – heatwave at the central market
    "Extreme heat warning near the central market - 40 plus degrees and no shade. "
    "Around 200 people are struggling. Urgent help needed immediately.",
    # Scenario 3 – water shortage, north district
    "Water supply cut off in the north district for 3 days. 50 families are "
    "severely dehydrated. Children are collapsing.",
    # Scenario 4 – power outage downtown
    "Blackout in downtown since last night. Hospitals are running on generators "
    "but residential blocks have no power. Critical situation.",
    # Scenario 5 – crowd crush at east station
    "Massive crowd crush at the east railway station exit. People are being "
    "trampled. This is an emergency - multiple injuries reported.",
]


if __name__ == "__main__":
    import uvicorn

    print("=" * 72)
    print("  PROJECT SKYPAD - AOS Stage 2 Routing Layer  (Demo Scenarios)")
    print("=" * 72)
    print(f"  GCP Project  : {GCP_PROJECT}")
    print(f"  Model        : {GEMINI_MODEL}")
    is_live = GCP_PROJECT != "your-gcp-project-id"
    print(f"  Mode         : {'LIVE (Vertex AI)' if is_live else 'DEMO (rule-based fallback)'}")
    print("=" * 72)

    for i, text in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n{'-' * 72}")
        print(f"  SCENARIO {i}")
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  Input : {preview!r}")
        print(f"{'-' * 72}")

        r = extract_crisis_and_route(text)

        print(f"  Crisis type     : {r.crisis_intent.crisis_type.value}")
        print(f"  Severity        : {r.crisis_intent.severity.value}  "
              f"(score={r.crisis_intent.severity_score:.2f})")
        print(f"  Required asset  : {r.crisis_intent.required_asset}")
        print(f"  Location cues   : {r.crisis_intent.location_cues}")
        print(f"  Urgency words   : {r.crisis_intent.urgency_keywords}")
        print(f"  Affected count  : {r.crisis_intent.affected_count}")
        print(f"  Reasoning       : {r.crisis_intent.reasoning}")
        print(f"  Dispatch (X,Y,Z): ({r.dispatch_coordinate.x}, "
              f"{r.dispatch_coordinate.y}, {r.dispatch_coordinate.z})")
        print(f"  Priority rank   : #{r.priority_rank}")
        print(f"  Transit time    : {r.estimated_transit_s} s")
        print(f"  Routing mode    : {r.routing_mode}")

    print(f"\n{'=' * 72}")
    print("  All demo scenarios completed successfully.")
    print("  Starting FastAPI routing server on http://127.0.0.1:8001")
    print("  API docs: http://127.0.0.1:8001/docs")
    print("=" * 72 + "\n")

    uvicorn.run(router_app, host="0.0.0.0", port=8001, log_level="info")
