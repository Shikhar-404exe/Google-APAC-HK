
<h1 align="center">
  🛸 PROJECT SKYPAD
</h1>

<h3 align="center">
  <i>Acoustic Operating System — AI-Powered Crisis Response via Sound</i>
</h3>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img alt="Google Cloud Run" src="https://img.shields.io/badge/Cloud%20Run-Deployed-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white"/>
  <img alt="Vertex AI" src="https://img.shields.io/badge/Vertex%20AI-Gemini%202.0-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white"/>
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<div align="center">
  <sub><i>Built for Google APAC Hackathon 2025 • Kinetic Oasis Track</i></sub>
</div>

<br/>

<p align="center">
  <a href="#-vision">Vision</a> •
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-local-development">Local Dev</a> •
  <a href="#-deployment">Deployment</a>
</p>

---

<br/>
## 💫 Vision

> **What if relief could fly, guided only by sound and AI?**

Urban crises strike fast — heatwaves, medical emergencies, water shortages. Traditional infrastructure is slow, expensive, land-hungry. **Project Skypad** removes the constraint of land entirely.

We built an **Acoustic Operating System (AOS)** — an AI brain that uses scaled acoustic levitation principles to hover, stabilize, and route lightweight utility blocks through mid-air directly to communities in need. No roads. No traffic. No delay.

This is the **Digital Twin Simulation** — a high-fidelity real-time control interface proving the intelligence behind the vision.

---

## 🎯 Live Demo

<table align="center">
  <tr>
    <td align="center"><b>🚀 Deployed Cloud Run Service</b></td>
    <td align="center">
      <a href="https://skypad-backend-78365051005.us-central1.run.app">
        <code>skypad-backend-78365051005.us-central1.run.app</code>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>📋 API Documentation</b></td>
    <td align="center">
      <a href="https://skypad-backend-78365051005.us-central1.run.app/docs">
        <code>/docs</code> (Swagger UI)
      </a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>🔄 WebSocket Stream</b></td>
    <td align="center"><code>wss://skypad-backend-78365051005.us-central1.run.app/ws</code></td>
  </tr>
  <tr>
    <td align="center"><b>❤️ Health Check</b></td>
    <td align="center">
      <a href="https://skypad-backend-78365051005.us-central1.run.app/health">
        <code>/health</code>
      </a>
    </td>
  </tr>
</table>

---

## 🏗 Architecture

```
                    ┌──────────────────────────┐
                    │  🌐 COMMUNITY DISTRESS    │
                    │  (Text / Audio / Alerts)  │
                    └──────────┬───────────────┘
                               │ POST /api/trigger/{crisis}
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                ☁️  CLOUD RUN (FastAPI Backend)               │
│                                                              │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐ │
│  │  🧠 Stage 2  │  │  ⚙️  Stage 1   │  │  📡  WebSocket   │ │
│  │  AI Routing  │──▶  Physics Engine │──▶  State Stream    │ │
│  │  (Gemini)    │  │  (PD Control)  │  │  @ 20 Hz         │ │
│  └──────────────┘  └────────────────┘  └────────┬────────┘ │
│                                                  │          │
└──────────────────────────────────────────────────┼──────────┘
                                                   │ ws://
                                                   ▼
┌────────────────────────────────────────────────────────────┐
│  🖥️  DIGITAL TWIN DASHBOARD (Canvas + WebSocket)           │
│  · Real-time 3D block visualization                         │
│  · Acoustic field overlay & phase matrix                    │
│  · Crisis dispatch buttons · Wind simulation                │
│  · Telemetry panel · Target tracking                        │
└────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Crisis Triggered** — A community distress signal arrives (or you click a button)
2. **AI Reasoning** — Stage 2 uses **Gemini (Vertex AI)** to extract crisis type, severity, location cues, and dispatch coordinates
3. **Physics Computation** — Stage 1 runs a **PD controller** with wind drag compensation, geofencing, velocity clamping
4. **Real-time Streaming** — The backend broadcasts state at **20 Hz** over WebSocket
5. **Digital Twin** — The dashboard renders the levitating block, acoustic nodes, wind particles, and telemetry

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Runtime** | 🐍 Python 3.12 | Core logic & computation |
| **API Framework** | ⚡ FastAPI + Uvicorn | REST + WebSocket endpoints |
| **AI Engine** | 🧠 Vertex AI (Gemini 2.0 Flash) | Crisis intent extraction & spatial routing |
| **Physics** | 📐 NumPy | Vector math, PD control, acoustic modeling |
| **Infrastructure** | 🐳 Docker + ☁️ Cloud Run | Containerized, auto-scaling serverless deploy |
| **Registry** | 📦 Artifact Registry | Docker image storage |
| **Frontend** | 🎨 Vanilla JS + Canvas | Real-time digital twin dashboard |
| **Real-time** | 🔌 WebSockets | 20 Hz state streaming |

---

## 📡 API Reference

### 🎯 Crisis Dispatch

```http
POST /api/trigger/{crisis}
```

| Crisis Key | Description | Asset |
|-----------|-------------|-------|
| `MEDICAL` | Medical emergency | `medical_kit` |
| `HEATWAVE` | Extreme heat alert | `shade_sail` |
| `WATER_SHORTAGE` | Water supply cut | `water_unit` |
| `POWER_OUTAGE` | Blackout event | `power_cell` |
| `CROWD_CRUSH` | Crowd stampede | `shade_sail` |

**Response:**
```json
{
  "status": "dispatched",
  "crisis": "HEATWAVE",
  "asset": "shade_sail",
  "target": [45.0, 50.0, 11.5],
  "priority": 1
}
```

### 🌬️ Wind Control

```http
POST /api/wind
Content-Type: application/json

{"x": 12.0, "y": 3.0, "z": 0.0}
```

> ⚠️ Wind speed ≥ 15 m/s triggers **EMERGENCY_DOCK** mode — the block safely descends to ground anchors.

### 🔄 Real-time State

```http
GET /api/state       # Full simulation snapshot
GET /api/reset       # Return block to staging pad
GET /health          # Liveness probe
WS  /ws              # 20 Hz state stream
```

---

## 🚀 Local Development

### Prerequisites

- Python 3.12+
- Docker (optional)

### Run from source

```bash
# Clone the repository
git clone https://github.com/Shikhar-404exe/Google-APAC-HK.git
cd "Google apac adition hk 1"

# Install dependencies
pip install -r requirements.txt

# Launch the backend
python app.py
```

Open **http://127.0.0.1:8000** — the digital twin dashboard loads immediately.

### Run with Docker

```bash
docker build -t skypad-backend .
docker run -p 8080:8080 skypad-backend
```

---

## ☁️ Deployment

This project is deployed on **Google Cloud Run** using the following pipeline:

```bash
# 1. Build & push to Artifact Registry
docker build -t us-central1-docker.pkg.dev/august-monolith-501613-g6/skypad-repo/skypad-backend:latest .
docker push us-central1-docker.pkg.dev/august-monolith-501613-g6/skypad-repo/skypad-backend:latest

# 2. Deploy to Cloud Run
gcloud run deploy skypad-backend \
  --image=us-central1-docker.pkg.dev/august-monolith-501613-g6/skypad-repo/skypad-backend:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --port=8080
```

The service auto-scales to zero when idle and handles traffic spikes seamlessly.

---

## 🎮 Dashboard Features

- **🎨 Real-time Canvas** — Levitating block with smooth client-side interpolation
- **🌆 City Skyline** — Procedural buildings with lit windows, stars, street view
- **🌀 Wind Particles** — Streaking particles that visualize wind direction & speed
- **🔊 Acoustic Node Grid** — 8 transducer nodes with phase-shift glow effects
- **📊 Telemetry Panel** — Power draw, correction vectors, gravity/drag forces
- **🚨 Crisis UI** — Colored buttons with active-state pulsing animations
- **💬 AI Reasoning** — See Gemini's classification & routing rationale live

---

## 🧪 Crisis Scenarios

| Scenario | Description |
|----------|-------------|
| 🏥 **Medical Emergency** | Clinic overcrowded, people fainting. Dispatches medical kit. |
| 🌡️ **Heatwave Alert** | 40°C+ at central market, 200 people struggling. Dispatches shade sail. |
| 💧 **Water Shortage** | North district cut off for 3 days, children collapsing. Dispatches water unit. |
| ⚡ **Power Outage** | Downtown blackout, hospitals on generator. Dispatches power cell. |
| 👥 **Crowd Crush** | East station stampede, multiple injuries. Dispatches shade sail. |

---

## 📁 Project Structure

```
├── app.py           # Unified FastAPI backend (Stage 1 + 2 + 3)
├── engine.py        # AOS Physics Engine (Stage 1)
├── routing.py       # AI Routing Layer (Stage 2)
├── index.html       # Digital Twin Dashboard (Stage 4)
├── requirements.txt # Python dependencies
├── Dockerfile       # Production container definition
├── .dockerignore    # Docker build exclusions
├── AOS_RULES.md     # Safety constraints & system prompt
├── PROJECT_OVERVIEW.md
└── TECHNOLOGY.md
```

---

<h2 align="center">✨ The Future of Crisis Response is Acoustic ✨</h2>

<p align="center">
  <sub>
    Built with ❤️ for the Google APAC Hackathon 2025<br/>
    Team: Shikhar • Project Skypad (Kinetic Oasis)
  </sub>
</p>

<p align="center">
  <a href="https://github.com/Shikhar-404exe/Google-APAC-HK">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github"/>
  </a>
</p>
