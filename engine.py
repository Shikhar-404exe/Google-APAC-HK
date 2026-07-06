"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PROJECT SKYPAD – STAGE 1: AOS CORE PHYSICS & MATRIX ENGINE          ║
║                              engine.py                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Acoustic Operating System (AOS) backend physics engine.

Models a 0.5 kg utility block levitating in a 3D coordinate space via an
acoustic transducer node grid. Calculates corrective force vectors to hold
or move the block smoothly within defined safety envelopes.

Architecture role:
    [Telemetry / API call]
         │
         ▼
    [FastAPI endpoint]  →  calculate_stabilization()  →  AOSResponse (JSON)
         │
         ▼
    [3D Visualiser / Frontend Dashboard]

AOS Safety Constraints (from AOS_RULES.md):
  • Z_min = 3.5 m  │  Z_max = 12.0 m
  • Radial safety buffer = 1.5 m around obstacles
  • V_max = 2.0 m/s
  • Wind > 15 m/s → EMERGENCY_DOCK mode
"""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL & AOS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_MASS_KG: float = 0.5          # Mass of the utility block (kg)
GRAVITY_MS2: float = 9.81           # Gravitational acceleration (m/s²)

# AOS geofencing boundaries (metres)
Z_MIN: float = 3.5                  # Minimum operational height above ground
Z_MAX: float = 12.0                 # Maximum operational height above ground
OBSTACLE_BUFFER_M: float = 1.5     # Hard safety bubble radius around obstacles

# Kinetic constraints
V_MAX_MS: float = 2.0               # Maximum block velocity (m/s)
WIND_EMERGENCY_THRESHOLD_MS: float = 15.0  # Wind speed that triggers emergency dock

# Acoustic model parameters
ACOUSTIC_NODES: int = 8             # Number of transducer nodes in the 3D rig
ACOUSTIC_MAX_POWER_PCT: float = 100.0
ACOUSTIC_BASE_LIFT_FORCE_N: float = BLOCK_MASS_KG * GRAVITY_MS2  # Hover equilibrium

# Air drag coefficient (simplified; unitless, tuned for cm-scale block)
DRAG_COEFFICIENT: float = 0.47      # Sphere-like object approximation
AIR_DENSITY_KGM3: float = 1.225    # Standard sea-level air density (kg/m³)
BLOCK_CROSS_SECTION_M2: float = 0.004  # ~4 cm² frontal area (lightweight block)

# PID-like controller gains for smooth corrections
KP: float = 1.2   # Proportional gain  – how hard to push per metre of error
KD: float = 0.4   # Derivative  gain   – damping to avoid overshoot

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA MODELS  (Pydantic – strict JSON I/O, no prose)
# ─────────────────────────────────────────────────────────────────────────────

class Vec3(BaseModel):
    """Immutable 3-axis vector [x, y, z] in metres or m/s."""
    x: float = Field(..., description="X-axis component (metres or m/s)")
    y: float = Field(..., description="Y-axis component (metres or m/s)")
    z: float = Field(..., description="Z-axis component (metres or m/s)")

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.to_np()))


class SystemStatus(str, Enum):
    STABLE          = "STABLE"           # Block is holding position; minimal correction
    COMPENSATION    = "COMPENSATION"     # Active correction in progress (wind / drift)
    EMERGENCY_DOCK  = "EMERGENCY_DOCK"   # Wind > 15 m/s; lowering to ground anchors


class StabilizationRequest(BaseModel):
    """Payload for a single stabilisation calculation cycle."""
    current_xyz: Vec3 = Field(..., description="Current block position (m)")
    target_xyz:  Vec3 = Field(..., description="Desired target position (m)")
    wind_vector: Vec3 = Field(
        default=Vec3(x=0.0, y=0.0, z=0.0),
        description="Instantaneous wind velocity vector (m/s)"
    )
    current_velocity: Vec3 = Field(
        default=Vec3(x=0.0, y=0.0, z=0.0),
        description="Current block velocity vector (m/s) for derivative damping"
    )


class AOSResponse(BaseModel):
    """
    Strict JSON data structure emitted to the hardware / visualiser layer.
    AOS_RULES.md section 3 mandates this format – no prose, only matrix values.
    """
    system_status:               SystemStatus
    target_coordinates:          Vec3
    correction_vector_ms2:       Vec3   # Net corrective acceleration (m/s²)
    acoustic_field_power_pct:    float  # 0–100 % aggregate transducer power draw
    phase_shift_matrix_delta:    List[float]  # Per-node phase delta (radians, 8 nodes)
    gravity_compensation_N:      float  # Static lift force applied (Newtons)
    wind_drag_force_N:           float  # Magnitude of drag force from wind (N)
    position_error_m:            float  # Euclidean distance from target (m)
    safe_velocity_vector_ms:     Vec3   # Clamped velocity command to hardware
    timestamp_utc:               float  # Unix timestamp for telemetry logging
    notes:                       Optional[str] = None  # Non-prose flags/warnings only


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ACOUSTIC NODE GRID  –  MATHEMATICAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_acoustic_node_grid(
    center: np.ndarray,
    radius_m: float = 0.25
) -> np.ndarray:
    """
    Return the 3D positions of the 8 acoustic transducer nodes arranged as a
    cube of side `radius_m * 2` centred around `center`.

    Each node is one vertex of the bounding cube – this models a real-world
    ultrasonic phased-array rig where 8 transducers surround the levitation
    chamber in a cubic arrangement.

    Args:
        center:   Centre of the levitation field (block position), shape (3,)
        radius_m: Half-edge of the surrounding transducer cube (metres)

    Returns:
        node_positions: (8, 3) ndarray of node world-space coordinates
    """
    offsets = np.array([
        [-1, -1, -1], [+1, -1, -1],
        [-1, +1, -1], [+1, +1, -1],
        [-1, -1, +1], [+1, -1, +1],
        [-1, +1, +1], [+1, +1, +1],
    ], dtype=float) * radius_m

    return center + offsets  # broadcast: (8, 3)


def compute_phase_shifts(
    node_positions: np.ndarray,
    target_position: np.ndarray,
    correction_vector: np.ndarray,
) -> List[float]:
    """
    Compute the acoustic phase shift (delta_phi, radians) required at each
    transducer node to steer the acoustic radiation-pressure focal point toward
    `target_position` with additional bias in the `correction_vector` direction.

    Physics basis:
        Acoustic radiation pressure P_rad = I / c
        where I = acoustic intensity (W/m²), c = speed of sound.

        Phase steering: delta_phi_i = (2*pi / lambda) * (d_i · n_correction)
        where d_i = vector from node_i to target,
              n_correction = unit vector of required correction force.

    For the simulation we use a simplified linear model: the phase delta is
    proportional to the projection of the node→target vector onto the
    correction direction.

    Args:
        node_positions:    (8, 3) node world positions
        target_position:   (3,)  focal point (block target)
        correction_vector: (3,)  net desired force direction

    Returns:
        List of 8 phase delta values in radians (one per node)
    """
    SPEED_OF_SOUND_MS = 343.0   # m/s at ~20 degrees C
    FREQ_HZ = 40_000.0          # 40 kHz – typical acoustic levitation frequency
    WAVELENGTH_M = SPEED_OF_SOUND_MS / FREQ_HZ  # ~8.575 mm

    k = (2 * math.pi) / WAVELENGTH_M  # Wave number (rad/m)

    # Unit vector of the required correction (direction to push the block)
    corr_norm = np.linalg.norm(correction_vector)
    if corr_norm < 1e-9:
        # No correction needed – all nodes emit in phase (holding field)
        return [0.0] * ACOUSTIC_NODES

    corr_unit = correction_vector / corr_norm

    phase_shifts: List[float] = []
    for node_pos in node_positions:
        d_vec = target_position - node_pos          # node to focal-point vector
        projection = float(np.dot(d_vec, corr_unit))  # scalar path-length bias
        delta_phi = k * projection                  # phase delta (radians)
        # Wrap to [-pi, pi] for hardware compatibility
        delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
        phase_shifts.append(round(delta_phi, 6))

    return phase_shifts


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CORE PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_gravity_compensation() -> float:
    """
    Return the static upward acoustic lift force required to counteract gravity.

        F_lift = m * g  =  0.5 kg * 9.81 m/s²  =  4.905 N

    This is the acoustic field baseline power commitment before any
    dynamic correction is applied.
    """
    return BLOCK_MASS_KG * GRAVITY_MS2  # Newtons


def compute_wind_drag(wind_vector: np.ndarray) -> np.ndarray:
    """
    Compute the aerodynamic drag force vector exerted on the block by wind.

    Using the drag equation:
        F_drag = 0.5 * rho_air * C_d * A * |v_wind|^2 * v_unit_wind

    where:
        rho_air  = air density (kg/m³)
        C_d      = drag coefficient (dimensionless)
        A        = frontal cross-sectional area (m²)
        v_wind   = wind velocity vector (m/s)

    Args:
        wind_vector: (3,) wind velocity in m/s

    Returns:
        drag_force: (3,) drag force vector in Newtons acting on the block
    """
    wind_speed = float(np.linalg.norm(wind_vector))
    if wind_speed < 1e-9:
        return np.zeros(3)

    wind_unit = wind_vector / wind_speed
    drag_magnitude = (
        0.5 * AIR_DENSITY_KGM3 * DRAG_COEFFICIENT
        * BLOCK_CROSS_SECTION_M2 * (wind_speed ** 2)
    )
    return drag_magnitude * wind_unit  # Direction follows wind direction


def clamp_velocity(velocity: np.ndarray) -> np.ndarray:
    """
    Enforce V_max kinetic constraint: if |velocity| > V_MAX_MS, scale it down
    to exactly V_MAX_MS while preserving direction.

    AOS_RULES.md section 2: Max block velocity = 2.0 m/s.
    """
    speed = float(np.linalg.norm(velocity))
    if speed > V_MAX_MS and speed > 1e-9:
        return velocity * (V_MAX_MS / speed)
    return velocity


def enforce_geofence(position: np.ndarray) -> np.ndarray:
    """
    Clamp block Z coordinate within [Z_MIN, Z_MAX] operational corridor.

    AOS_RULES.md section 1:
        Z_min = 3.5 m  (above pedestrian reach)
        Z_max = 12.0 m (below powerlines / low structures)
    """
    clamped = position.copy()
    clamped[2] = float(np.clip(clamped[2], Z_MIN, Z_MAX))
    return clamped


def determine_acoustic_power(position_error_m: float, wind_speed_ms: float) -> float:
    """
    Estimate aggregate acoustic transducer power as a percentage of maximum.

    Composed of:
        - Base hover power  : cancels gravity; always present (~33%)
        - Error correction  : scales with position error (further = more power)
        - Wind compensation : scales with wind drag load

    The result is clamped to [0, 100] %.
    """
    # Hover power: gravity compensation vs max force capacity (3x headroom)
    max_acoustic_force = 3.0 * ACOUSTIC_BASE_LIFT_FORCE_N
    hover_pct = (ACOUSTIC_BASE_LIFT_FORCE_N / max_acoustic_force) * 100.0  # ~33%

    # Error correction overhead (up to +30%)
    error_overhead_pct = min(30.0, position_error_m * 10.0)

    # Wind compensation overhead (up to +37%)
    wind_overhead_pct = min(37.0, wind_speed_ms * 2.0)

    total = hover_pct + error_overhead_pct + wind_overhead_pct
    return round(min(ACOUSTIC_MAX_POWER_PCT, max(0.0, total)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN STABILISATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_stabilization(
    current_xyz: Vec3,
    target_xyz: Vec3,
    wind_vector: Vec3,
    current_velocity: Vec3 = Vec3(x=0.0, y=0.0, z=0.0),
) -> AOSResponse:
    """
    Calculate the acoustic vector corrections required to hold or smoothly move
    the 0.5 kg utility block from `current_xyz` to `target_xyz` against
    `wind_vector`.

    Physics pipeline:
        1. Enforce geofencing on target Z coordinate
        2. Check emergency wind threshold → EMERGENCY_DOCK if wind >= 15 m/s
        3. Compute position error vector  → PD proportional correction
        4. Compute wind drag force        → counter-force correction
        5. Sum corrections into net acceleration vector
        6. Clamp velocity command to V_max
        7. Generate acoustic phase-shift matrix for 8 transducer nodes
        8. Estimate aggregate power draw percentage
        9. Return structured AOSResponse

    Args:
        current_xyz:      Current block world position (metres)
        target_xyz:       Desired block position (metres)
        wind_vector:      Instantaneous wind velocity (m/s)
        current_velocity: Current block velocity (m/s) for derivative damping

    Returns:
        AOSResponse: Strict JSON-compatible data structure for the hardware layer
    """

    # Convert Pydantic models to numpy arrays for vector math
    pos_cur   = current_xyz.to_np()
    pos_tgt   = target_xyz.to_np()
    vel_cur   = current_velocity.to_np()
    wind_np   = wind_vector.to_np()

    # ── Step 1: Enforce geofence on target ────────────────────────────────────
    pos_tgt = enforce_geofence(pos_tgt)

    # ── Step 2: Wind emergency check ──────────────────────────────────────────
    wind_speed = float(np.linalg.norm(wind_np))
    if wind_speed >= WIND_EMERGENCY_THRESHOLD_MS:
        # AOS_RULES.md section 2: Slowly lower block to ground anchors (Z_MIN)
        safe_dock_target = np.array([pos_cur[0], pos_cur[1], Z_MIN])
        dock_correction  = safe_dock_target - pos_cur
        safe_vel_np      = clamp_velocity(dock_correction * 0.3)  # gentle descent

        drag_force_np    = compute_wind_drag(wind_np)
        drag_mag         = float(np.linalg.norm(drag_force_np))
        gravity_comp_N   = compute_gravity_compensation()

        node_positions   = build_acoustic_node_grid(pos_cur)
        phase_deltas     = compute_phase_shifts(
            node_positions, safe_dock_target, dock_correction
        )

        return AOSResponse(
            system_status            = SystemStatus.EMERGENCY_DOCK,
            target_coordinates       = Vec3(
                x=safe_dock_target[0],
                y=safe_dock_target[1],
                z=safe_dock_target[2]
            ),
            correction_vector_ms2    = Vec3(
                x=round(dock_correction[0], 6),
                y=round(dock_correction[1], 6),
                z=round(dock_correction[2], 6),
            ),
            acoustic_field_power_pct = 95.0,   # Near-max to fight emergency wind
            phase_shift_matrix_delta = phase_deltas,
            gravity_compensation_N   = round(gravity_comp_N, 4),
            wind_drag_force_N        = round(drag_mag, 4),
            position_error_m         = round(float(np.linalg.norm(dock_correction)), 4),
            safe_velocity_vector_ms  = Vec3(
                x=round(safe_vel_np[0], 6),
                y=round(safe_vel_np[1], 6),
                z=round(safe_vel_np[2], 6),
            ),
            timestamp_utc            = time.time(),
            notes                    = (
                f"EMERGENCY_DOCK: wind={wind_speed:.2f} m/s "
                f">= threshold={WIND_EMERGENCY_THRESHOLD_MS} m/s. "
                "Initiating controlled descent to ground anchors."
            ),
        )

    # ── Step 3: Position error & PD correction ────────────────────────────────
    # Error vector (metres): direction and distance to target
    error_vec      = pos_tgt - pos_cur
    position_error = float(np.linalg.norm(error_vec))

    # PD control law:  a_corr = Kp * error - Kd * velocity
    # Proportional term drives block toward target;
    # Derivative  term damps velocity to prevent overshoot.
    proportional   = KP * error_vec
    derivative     = KD * vel_cur   # dampen current motion
    pd_correction  = proportional - derivative  # net corrective acceleration (m/s²)

    # ── Step 4: Wind drag counter-force ───────────────────────────────────────
    drag_force_np  = compute_wind_drag(wind_np)
    drag_mag       = float(np.linalg.norm(drag_force_np))

    # Counter-drag acceleration: apply equal and opposite force, divide by mass
    #   F = m * a  →  a = F / m
    wind_counter_acc = -drag_force_np / BLOCK_MASS_KG  # m/s² (opposes wind drag)

    # ── Step 5: Net correction vector ─────────────────────────────────────────
    net_correction = pd_correction + wind_counter_acc

    # ── Step 6: Velocity command & clamping ───────────────────────────────────
    # Convert corrective acceleration to velocity command (dt = 50ms control tick)
    DT = 0.05  # Control tick period (seconds)
    raw_velocity   = vel_cur + net_correction * DT
    safe_velocity  = clamp_velocity(raw_velocity)

    # ── Step 7: Acoustic phase-shift matrix ───────────────────────────────────
    node_positions = build_acoustic_node_grid(pos_cur)
    phase_deltas   = compute_phase_shifts(node_positions, pos_tgt, net_correction)

    # ── Step 8: Power estimate ────────────────────────────────────────────────
    power_pct      = determine_acoustic_power(position_error, wind_speed)
    gravity_comp_N = compute_gravity_compensation()

    # ── Step 9: Determine system status ───────────────────────────────────────
    #  STABLE       → error < 5 cm and wind < 3 m/s (calm hover)
    #  COMPENSATION → active correction required
    if position_error < 0.05 and wind_speed < 3.0:
        status = SystemStatus.STABLE
    else:
        status = SystemStatus.COMPENSATION

    return AOSResponse(
        system_status            = status,
        target_coordinates       = Vec3(
            x=round(pos_tgt[0], 6),
            y=round(pos_tgt[1], 6),
            z=round(pos_tgt[2], 6),
        ),
        correction_vector_ms2    = Vec3(
            x=round(net_correction[0], 6),
            y=round(net_correction[1], 6),
            z=round(net_correction[2], 6),
        ),
        acoustic_field_power_pct = power_pct,
        phase_shift_matrix_delta = phase_deltas,
        gravity_compensation_N   = round(gravity_comp_N, 4),
        wind_drag_force_N        = round(drag_mag, 4),
        position_error_m         = round(position_error, 6),
        safe_velocity_vector_ms  = Vec3(
            x=round(safe_velocity[0], 6),
            y=round(safe_velocity[1], 6),
            z=round(safe_velocity[2], 6),
        ),
        timestamp_utc            = time.time(),
        notes                    = None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FASTAPI  APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Project Skypad – AOS Physics Engine",
    description = (
        "Stage 1 Core Physics & Matrix Engine. "
        "Simulates acoustic levitation stabilisation of a 0.5 kg utility block "
        "in a 3D grid space. Enforces AOS safety boundaries and outputs strict "
        "JSON vector arrays for transducer control."
    ),
    version     = "1.0.0",
)


@app.get("/health", tags=["System"])
def health_check() -> dict:
    """
    Basic liveness probe for Cloud Run / load-balancer health checks.
    Returns engine constants so upstream services can verify configuration.
    """
    return {
        "status":         "online",
        "engine":         "AOS Physics Engine v1.0.0",
        "stage":          1,
        "block_mass_kg":  BLOCK_MASS_KG,
        "z_min_m":        Z_MIN,
        "z_max_m":        Z_MAX,
        "v_max_ms":       V_MAX_MS,
        "acoustic_nodes": ACOUSTIC_NODES,
        "timestamp_utc":  time.time(),
    }


@app.post("/stabilize", response_model=AOSResponse, tags=["AOS Control"])
def stabilize(request: StabilizationRequest) -> AOSResponse:
    """
    Primary stabilisation endpoint.

    Accepts current position, target position, and wind vector.
    Returns a fully computed AOSResponse with corrective vectors,
    phase-shift matrix, power draw, and system status.

    Example request body:
    {
        "current_xyz": {"x": 5.0, "y": 5.0, "z": 6.0},
        "target_xyz":  {"x": 5.0, "y": 5.0, "z": 6.0},
        "wind_vector": {"x": 3.0, "y": 0.5, "z": 0.0},
        "current_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}
    }
    """
    try:
        return calculate_stabilization(
            current_xyz      = request.current_xyz,
            target_xyz       = request.target_xyz,
            wind_vector      = request.wind_vector,
            current_velocity = request.current_velocity,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/move", response_model=AOSResponse, tags=["AOS Control"])
def move_block(request: StabilizationRequest) -> AOSResponse:
    """
    Trajectory correction endpoint – semantically signals an active relocation
    command rather than a hold-in-place command. Future stages will accept
    waypoint lists here. Geofence is automatically enforced on target Z.
    """
    return stabilize(request)


@app.get("/grid/status", tags=["Diagnostics"])
def grid_status() -> dict:
    """
    Returns the current 3D acoustic node grid layout centred at the
    nominal hover midpoint (0, 0, Z_mid) for diagnostic/visualiser use.
    """
    z_mid   = (Z_MIN + Z_MAX) / 2.0  # 7.75 m
    center  = np.array([0.0, 0.0, z_mid])
    nodes   = build_acoustic_node_grid(center).tolist()

    return {
        "grid_center":     {"x": 0.0, "y": 0.0, "z": z_mid},
        "node_count":      ACOUSTIC_NODES,
        "node_positions":  [{"id": i, "x": n[0], "y": n[1], "z": n[2]}
                            for i, n in enumerate(nodes)],
        "frequency_hz":    40_000,
        "wavelength_mm":   round(343.0 / 40_000 * 1000, 3),
        "timestamp_utc":   time.time(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LOCAL DEV ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Quick sanity-check demonstration in the console
    print("=" * 70)
    print("  PROJECT SKYPAD – AOS Stage 1 Engine  (Sanity Check)")
    print("=" * 70)

    # --- Scenario A: Block holding steady, mild breeze ----------------------
    scenario_a = calculate_stabilization(
        current_xyz      = Vec3(x=5.0,  y=5.0,  z=6.5),
        target_xyz       = Vec3(x=5.0,  y=5.0,  z=6.5),
        wind_vector      = Vec3(x=2.5,  y=0.0,  z=0.0),
        current_velocity = Vec3(x=0.0,  y=0.0,  z=0.0),
    )
    print("\n[A] Holding position in mild breeze (2.5 m/s):")
    print(f"    Status            : {scenario_a.system_status}")
    print(f"    Position error    : {scenario_a.position_error_m} m")
    print(f"    Acoustic power    : {scenario_a.acoustic_field_power_pct} %")
    print(f"    Correction vector : {scenario_a.correction_vector_ms2}")
    print(f"    Phase shifts      : {scenario_a.phase_shift_matrix_delta}")

    # --- Scenario B: Block moving to a new location, moderate wind ----------
    scenario_b = calculate_stabilization(
        current_xyz      = Vec3(x=3.0,  y=2.0,  z=5.0),
        target_xyz       = Vec3(x=8.0,  y=7.0,  z=9.0),
        wind_vector      = Vec3(x=5.0,  y=2.0,  z=0.0),
        current_velocity = Vec3(x=1.0,  y=0.5,  z=0.2),
    )
    print("\n[B] Moving block to new target, moderate wind (5.4 m/s):")
    print(f"    Status            : {scenario_b.system_status}")
    print(f"    Position error    : {scenario_b.position_error_m} m")
    print(f"    Safe velocity     : {scenario_b.safe_velocity_vector_ms}")
    print(f"    Wind drag force   : {scenario_b.wind_drag_force_N} N")
    print(f"    Acoustic power    : {scenario_b.acoustic_field_power_pct} %")

    # --- Scenario C: Emergency wind event ----------------------------------
    scenario_c = calculate_stabilization(
        current_xyz      = Vec3(x=5.0,  y=5.0,  z=8.0),
        target_xyz       = Vec3(x=6.0,  y=6.0,  z=8.0),
        wind_vector      = Vec3(x=18.0, y=4.0,  z=0.0),  # 18.4 m/s – EMERGENCY
    )
    print("\n[C] Emergency wind event (18.4 m/s):")
    print(f"    Status            : {scenario_c.system_status}")
    print(f"    Emergency target  : {scenario_c.target_coordinates}")
    print(f"    Notes             : {scenario_c.notes}")
    print("=" * 70)

    # Launch the API server
    print("\nStarting FastAPI server at http://127.0.0.1:8000")
    print("Docs available at  http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
