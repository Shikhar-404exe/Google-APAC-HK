"""
PROJECT SKYPAD - STAGE 3: UNIFIED CLOUD BACKEND API
app.py

Combines Stage 1 (engine.py physics) and Stage 2 (routing.py AI routing) into
a single cohesive FastAPI service that:
  - Runs a real-time physics simulation tick at 20 Hz
  - Streams live block state via WebSocket to the Stage 4 frontend
  - Exposes REST endpoints for crisis triggers, wind control, and reset
  - Serves the Stage 4 dashboard (index.html) at the root URL

Run:
    python app.py
    Open:  http://127.0.0.1:8000
    Docs:  http://127.0.0.1:8000/docs
    WS:    ws://127.0.0.1:8000/ws
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure same-directory imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Stage 1: Physics engine
from engine import (
    Vec3, calculate_stabilization,
    V_MAX_MS, Z_MIN, Z_MAX, ACOUSTIC_NODES,
)

# Stage 2: AI routing
from routing import (
    extract_crisis_and_route,
    _demo_rule_based_extraction,
    _dict_to_crisis_intent,
    _SEVERITY_Z_MAP,
    GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX, GRID_Z_MIN, GRID_Z_MAX,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skypad.app")

# ---------------------------------------------------------------------------
# 1.  GLOBAL SIMULATION STATE
# ---------------------------------------------------------------------------
# Single source of truth shared between the physics tick and all WS clients.
# All positions in metres; velocity in m/s; forces in Newtons.

SIM: dict = {
    "current":     [5.0,  50.0, 3.5],    # block world position [x, y, z]
    "target":      [5.0,  50.0, 3.5],    # current dispatch target
    "wind":        [0.0,  0.0,  0.0],    # wind velocity vector
    "velocity":    [0.0,  0.0,  0.0],    # block velocity

    # AOS response fields (updated every physics tick)
    "status":      "STABLE",
    "power":       33.3,
    "error_m":     0.0,
    "correction":  [0.0, 0.0, 0.0],
    "phase_shifts":[0.0] * ACOUSTIC_NODES,
    "drag_N":      0.0,
    "gravity_N":   4.905,

    # Routing metadata
    "crisis":      "IDLE",
    "asset":       "none",
    "reasoning":   "",
    "priority":    10,
}

# Physics control loop period
TICK_HZ   = 20       # ticks per second
DT        = 1.0 / TICK_HZ

# ---------------------------------------------------------------------------
# 2.  CRISIS SCENARIO LIBRARY
# ---------------------------------------------------------------------------
# Each entry provides:
#   "text"   – community report (fed to routing.py)
#   "preset" – fallback target if routing is in demo mode (already known coords)

CRISIS_PRESETS: dict[str, dict] = {
    "MEDICAL": {
        "text":   "The medical clinic on 4th street is overcrowded and people are fainting outside in the sun.",
        "preset": [40.0, 50.0, 9.0],
    },
    "HEATWAVE": {
        "text":   "Extreme heat warning near the central market - 40+ degrees, no shade. Around 200 people struggling. Urgent immediately.",
        "preset": [45.0, 50.0, 11.5],
    },
    "WATER_SHORTAGE": {
        "text":   "Water supply cut off in the north district for 3 days. 50 families are severely dehydrated. Children are collapsing.",
        "preset": [50.0, 85.0, 9.0],
    },
    "POWER_OUTAGE": {
        "text":   "Blackout in downtown since last night. Hospitals on generators, residential blocks dark. Critical situation.",
        "preset": [50.0, 50.0, 11.5],
    },
    "CROWD_CRUSH": {
        "text":   "Massive crowd crush at the east railway station exit. People trampled. Emergency - multiple injuries.",
        "preset": [85.0, 50.0, 11.5],
    },
}

# ---------------------------------------------------------------------------
# 3.  WEBSOCKET CONNECTION MANAGER
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Tracks active WebSocket connections and handles broadcasts."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        log.info("WS client connected  total=%d", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        log.info("WS client disconnected  total=%d", len(self._clients))

    async def broadcast(self, message: str) -> None:
        """Send message to all connected clients; prune dead connections."""
        dead: set[WebSocket] = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    @property
    def count(self) -> int:
        return len(self._clients)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# 4.  PHYSICS TICK  (runs as an asyncio background task)
# ---------------------------------------------------------------------------

async def physics_tick_loop() -> None:
    """
    Runs at TICK_HZ Hz.  Each tick:
        1. Calls calculate_stabilization() from Stage 1
        2. Integrates velocity to update current block position
        3. Updates SIM dict
        4. Broadcasts serialised state to all WebSocket clients
    """
    log.info("Physics tick loop started  hz=%d  dt=%.3fs", TICK_HZ, DT)

    while True:
        await asyncio.sleep(DT)

        # Snapshot state (avoid mutation mid-tick)
        curr_xyz  = Vec3(x=SIM["current"][0],  y=SIM["current"][1],  z=SIM["current"][2])
        tgt_xyz   = Vec3(x=SIM["target"][0],   y=SIM["target"][1],   z=SIM["target"][2])
        wind_xyz  = Vec3(x=SIM["wind"][0],     y=SIM["wind"][1],     z=SIM["wind"][2])
        vel_xyz   = Vec3(x=SIM["velocity"][0], y=SIM["velocity"][1], z=SIM["velocity"][2])

        # Stage 1 physics calculation
        try:
            resp = calculate_stabilization(curr_xyz, tgt_xyz, wind_xyz, vel_xyz)
        except Exception as exc:
            log.error("Physics tick error: %s", exc)
            continue

        # Extract safe velocity from response
        sv = resp.safe_velocity_vector_ms
        new_vel = [sv.x, sv.y, sv.z]

        # Integrate position: x_new = x + v*dt
        new_pos = [
            max(GRID_X_MIN, min(GRID_X_MAX, SIM["current"][0] + sv.x * DT)),
            max(GRID_Y_MIN, min(GRID_Y_MAX, SIM["current"][1] + sv.y * DT)),
            max(GRID_Z_MIN, min(GRID_Z_MAX, SIM["current"][2] + sv.z * DT)),
        ]

        # Mutate SIM state
        SIM["current"]      = new_pos
        SIM["velocity"]     = new_vel
        SIM["status"]       = resp.system_status.value
        SIM["power"]        = resp.acoustic_field_power_pct
        SIM["error_m"]      = resp.position_error_m
        SIM["correction"]   = [
            resp.correction_vector_ms2.x,
            resp.correction_vector_ms2.y,
            resp.correction_vector_ms2.z,
        ]
        SIM["phase_shifts"] = resp.phase_shift_matrix_delta
        SIM["drag_N"]       = resp.wind_drag_force_N
        SIM["gravity_N"]    = resp.gravity_compensation_N

        # Broadcast if any clients connected
        if manager.count > 0:
            payload = {
                "current":      SIM["current"],
                "target":       SIM["target"],
                "wind":         SIM["wind"],
                "status":       SIM["status"],
                "power":        round(SIM["power"], 2),
                "error_m":      round(SIM["error_m"], 4),
                "correction":   [round(v, 4) for v in SIM["correction"]],
                "phase_shifts": [round(p, 4) for p in SIM["phase_shifts"]],
                "drag_N":       round(SIM["drag_N"], 4),
                "gravity_N":    round(SIM["gravity_N"], 4),
                "crisis":       SIM["crisis"],
                "asset":        SIM["asset"],
                "priority":     SIM["priority"],
                "reasoning":    SIM["reasoning"],
                "ts":           time.time(),
            }
            await manager.broadcast(json.dumps(payload))

# ---------------------------------------------------------------------------
# 5.  FASTAPI APP  +  LIFESPAN
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Start background physics loop on startup; clean up on shutdown."""
    task = asyncio.create_task(physics_tick_loop())
    log.info("Project Skypad backend started.")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    log.info("Project Skypad backend stopped.")


app = FastAPI(
    title       = "Project Skypad - AOS Unified Backend",
    description = (
        "Stage 3 cohesive backend. Integrates Stage 1 physics (engine.py) "
        "and Stage 2 AI routing (routing.py) with real-time WebSocket streaming "
        "for the Stage 4 digital twin dashboard."
    ),
    version     = "3.0.0",
    lifespan    = lifespan,
)

# CORS: allow the frontend (any origin during dev/hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 6.  FRONTEND  –  serve index.html at root
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the Stage 4 digital twin dashboard."""
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        return JSONResponse(
            {"error": "index.html not found. Run Stage 4 setup first."},
            status_code=404,
        )
    return FileResponse(str(html_path), media_type="text/html")


# ---------------------------------------------------------------------------
# 7.  WEBSOCKET ENDPOINT
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Real-time state stream.
    Clients receive a JSON state packet every physics tick (~50 ms).
    """
    await manager.connect(ws)
    try:
        # Send immediate snapshot on connect so the UI doesn't wait for first tick
        await ws.send_text(json.dumps({
            "current":      SIM["current"],
            "target":       SIM["target"],
            "wind":         SIM["wind"],
            "status":       SIM["status"],
            "power":        SIM["power"],
            "error_m":      SIM["error_m"],
            "correction":   SIM["correction"],
            "phase_shifts": SIM["phase_shifts"],
            "drag_N":       SIM["drag_N"],
            "gravity_N":    SIM["gravity_N"],
            "crisis":       SIM["crisis"],
            "asset":        SIM["asset"],
            "priority":     SIM["priority"],
            "reasoning":    SIM["reasoning"],
            "ts":           time.time(),
        }))
        # Keep alive (physics loop broadcasts; client just listens)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as exc:
        log.warning("WS error: %s", exc)
        manager.disconnect(ws)

# ---------------------------------------------------------------------------
# 8.  REST API ENDPOINTS
# ---------------------------------------------------------------------------

class WindRequest(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@app.post("/api/trigger/{crisis}", tags=["AOS Control"])
async def trigger_crisis(crisis: str) -> dict:
    """
    Trigger a crisis scenario.  The routing layer resolves the dispatch
    coordinate and the physics engine begins moving the block.

    Supported crisis types:
        MEDICAL | HEATWAVE | WATER_SHORTAGE | POWER_OUTAGE | CROWD_CRUSH
    """
    crisis_key = crisis.upper().replace("-", "_").replace(" ", "_")

    if crisis_key not in CRISIS_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crisis '{crisis_key}'. "
                   f"Valid: {list(CRISIS_PRESETS.keys())}",
        )

    scenario  = CRISIS_PRESETS[crisis_key]
    log.info("Crisis triggered  type=%s", crisis_key)

    # Use routing layer to extract intent and coordinates
    try:
        route = extract_crisis_and_route(scenario["text"])
        target = [
            route.dispatch_coordinate.x,
            route.dispatch_coordinate.y,
            route.dispatch_coordinate.z,
        ]
        SIM["crisis"]    = route.crisis_intent.crisis_type.value
        SIM["asset"]     = route.crisis_intent.required_asset
        SIM["priority"]  = route.priority_rank
        SIM["reasoning"] = route.crisis_intent.reasoning
    except Exception as exc:
        log.warning("Routing failed (%s); using preset coords.", exc)
        target = scenario["preset"]
        SIM["crisis"]    = crisis_key
        SIM["asset"]     = {
            "MEDICAL":        "medical_kit",
            "HEATWAVE":       "shade_sail",
            "WATER_SHORTAGE": "water_unit",
            "POWER_OUTAGE":   "power_cell",
            "CROWD_CRUSH":    "shade_sail",
        }.get(crisis_key, "medical_kit")
        SIM["priority"]  = 1
        SIM["reasoning"] = f"Preset dispatch for {crisis_key}."

    SIM["target"] = target
    log.info("Dispatch target set  target=%s", target)

    return {
        "status":   "dispatched",
        "crisis":   SIM["crisis"],
        "asset":    SIM["asset"],
        "target":   SIM["target"],
        "priority": SIM["priority"],
    }


@app.post("/api/wind", tags=["AOS Control"])
async def set_wind(body: WindRequest) -> dict:
    """
    Update the wind vector.  The physics engine immediately includes this
    in its drag compensation calculations.

    Wind speed >= 15 m/s triggers EMERGENCY_DOCK mode.
    """
    SIM["wind"] = [body.x, body.y, body.z]
    import math
    speed = math.sqrt(body.x**2 + body.y**2 + body.z**2)
    log.info("Wind updated  vector=%s  speed=%.2f m/s", SIM["wind"], speed)
    return {"status": "ok", "wind": SIM["wind"], "speed_ms": round(speed, 2)}


@app.get("/api/reset", tags=["AOS Control"])
async def reset() -> dict:
    """
    Return the block to the staging pad (X=5, Y=50, Z=3.5) and clear
    the active crisis.
    """
    SIM["target"]    = [5.0, 50.0, 3.5]
    SIM["wind"]      = [0.0, 0.0,  0.0]
    SIM["crisis"]    = "IDLE"
    SIM["asset"]     = "none"
    SIM["priority"]  = 10
    SIM["reasoning"] = ""
    log.info("System reset to staging.")
    return {"status": "reset", "target": SIM["target"]}


@app.get("/api/state", tags=["Diagnostics"])
def get_state() -> dict:
    """Return the current full simulation state (REST snapshot)."""
    return dict(SIM)


@app.get("/health", tags=["System"])
def health() -> dict:
    return {
        "status":     "online",
        "service":    "AOS Unified Backend v3.0.0",
        "stage":      3,
        "ws_clients": manager.count,
        "tick_hz":    TICK_HZ,
        "ts":         time.time(),
    }

# ---------------------------------------------------------------------------
# 9.  ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting Project Skypad AOS Backend on http://0.0.0.0:%d", port)
    log.info("Dashboard: http://127.0.0.1:%d", port)
    log.info("API docs:  http://127.0.0.1:%d/docs", port)
    log.info("WebSocket: ws://127.0.0.1:%d/ws", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
