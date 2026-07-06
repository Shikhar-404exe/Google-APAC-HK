# Project Skypad (Kinetic Oasis) – Project Overview

### Vision
Project Skypad addresses urban resource deserts and localized crises by removing the constraint of land. Instead of permanent, slow, expensive physical infrastructure, Skypad deploys an AI-controlled Acoustic Operating System (AOS) that uses scaled acoustic levitation principles to hover, stabilize, and route ultra-lightweight utility blocks (medical kits, shade sails, water filtration units) through mid-air directly to communities in need.

### The 5-Hour Digital Twin Hackathon Strategy
Because physical hardware deployment takes time, this project delivers the core intelligence: a high-fidelity **Digital Twin Simulation and Real-Time Acoustic Control Interface**. It demonstrates how AI can handle physics computations and natural language crisis routing simultaneously.

---

## Architecture & Data Flow
[Simulated Real-World Distress Data] (Spikes in Temp, Audio, Alerts)
│
▼ (Pushed via WebSockets / API)
[Cloud Run Python Backend] <───> [Gemini 1.5 Flash / Pro]
│                                  │
│ (Calculates Real-Time Physics)  │ (Extracts Intent & Coordinates)
▼                                  ▼
[Acoustic Operating System (AOS) Vector Output Grid]
│
▼ (Real-time Stream)
[Frontend Interactive 3D/2D Visualizer Dashboard]

---

## Project Development Stages (5-Hour Sprint)

### Stage 1: Core Physics & Matrix Engine (Hour 1)
* Set up the mathematical coordinate environment (3D matrix grid) modeling acoustic potential nodes.
* Write the physics math handling gravity ($F_g = mg$), wind resistance vector adjustments, and safety boundaries.

### Stage 2: Gemini Intent & Spatial Mapping Layer (Hour 2)
* Establish the Google Cloud Vertex AI connection.
* Configure the LLM reasoning loop to parse raw text/audio distress updates from a community and map them to localized targets $(X, Y, Z)$.

### Stage 3: The Cloud Backend API (Hour 3)
* Build a fast, lightweight FastAPI/Python service deployed to Google Cloud Run.
* Create endpoints for real-time telemetry streaming and event generation (e.g., triggering a "Heatwave Emergency").

### Stage 4: Frontend Digital Twin Dashboard (Hour 4)
* Build a clean, industrial-minimalist visual canvas mapping the physical street level and the floating blocks.
* Hook up WebSockets to show the blocks physically shifting and stabilizing dynamically against simulated wind.

### Stage 5: Validation, Polish & Pitch (Hour 5)
* Test safety override vectors (e.g., object dropping safely or dodging unexpected obstacles).
* Freeze code and prepare the demonstration walkthrough narrative.