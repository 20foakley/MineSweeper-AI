# MineSearcher v1.0

> **Neuro-Symbolic AI & Explainable AI Research Platform**  
> Master's Research Project — Rutgers University, Department of Computer Science

MineSearcher is a sophisticated research testbed for exploring the intersection of **probabilistic deep learning** and **deterministic spatial logic** within the context of autonomous hazard detection and battlefield mine-clearance strategies. The platform couples a hybrid Neuro-Symbolic agent with an Explainable AI (XAI) critique engine, enabling rigorous academic study of autonomous decision-making under uncertainty.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [1. Repository Structure](#1-repository-structure)
  - [2. Backend Setup](#2-backend-setup)
  - [3. Frontend Setup](#3-frontend-setup)
- [Running the Platform](#running-the-platform)
- [Research Capabilities](#research-capabilities)
- [Configuration](#configuration)
- [Project Status](#project-status)

---

## Overview

MineSearcher simulates autonomous mine-clearance missions to benchmark hybrid AI architectures. The core agent fuses a **3-layer Convolutional Neural Network (CNN)** for spatial intuition with a **Deterministic Constraint Satisfaction Solver** for rigorous logical deduction — a neuro-symbolic design that mirrors how expert human de-miners combine intuition with rule-based reasoning.

A post-mission XAI engine, powered by **Gemma 2 27B**, delivers natural language tactical critiques of agent behavior, making the system's internal logic interpretable for human researchers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MineSearcher Platform                 │
│                                                         │
│  ┌──────────────────┐       ┌────────────────────────┐  │
│  │   React Frontend │◄─────►│    FastAPI Backend      │  │
│  │   (Vite + JS)    │       │    (Python 3.12+)       │  │
│  └──────────────────┘       └────────────┬───────────┘  │
│                                          │               │
│              ┌───────────────────────────┼─────────┐     │
│              │                           │         │     │
│   ┌──────────▼──────┐   ┌───────────────▼──┐      │     │
│   │  CNN (3-Layer)  │   │  CSP Constraint  │      │     │
│   │  hazard_model   │   │     Solver        │      │     │
│   │     .h5         │   └──────────────────┘      │     │
│   └─────────────────┘                              │     │
│                                                    │     │
│              ┌─────────────────────────────────────┘     │
│              │                                           │
│   ┌──────────▼──────────────────────────────────────┐   │
│   │  XAI Engine — Gemma 2 27B (Google Gen AI)        │   │
│   │  Natural language post-mortems & tactical        │   │
│   │  critiques of mission telemetry                  │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

### 🧠 Neuro-Symbolic Agent
A hybrid architecture combining:
- **CNN (3-layer)** — learns spatial probability distributions over hazard locations from raw grid state
- **Constraint Satisfaction Solver** — enforces deterministic logical deductions (e.g., a cell guaranteed safe by neighbor constraints is never flagged as dangerous)

The two modules operate in a perception-reasoning pipeline, where CNN output probability maps are fed into the CSP solver as soft constraints.

### 🔍 Explainable AI (XAI) Engine
Powered by **Gemma 2 27B** via Google Generative AI. After each mission or batch run, the engine:
- Analyzes full mission telemetry
- Produces natural language post-mortems detailing agent decisions
- Issues tactical critiques identifying suboptimal moves and near-misses
- Supports iterative research on agent interpretability

### 📊 Batch Benchmarking
- Run up to **500 autonomous simulations simultaneously**
- Automatic grid categorization by complexity tier (`Easy`, `Medium`, `Hard`)
- Pattern detection to identify grid configurations that **maximize agent uncertainty**
- Aggregated statistics exported for downstream statistical analysis

### 🗺️ Interactive Heatmaps & Optimal Pathing
- Real-time visualization of the CNN's probability distributions as color-gradient heatmaps
- Dynamic **Optimal Path** overlay showing the highest-confidence safe-clearance route
- Step-by-step mission replay for manual analysis

### ⚙️ Battlefield Optimization Engine
- Continuous scoring of mission outcomes, categorized as `Easy` or `Hard` battlefield patterns
- Persistent storage of high-difficulty configurations for adversarial training research
- Strategic insight dashboard for analyzing efficient hazard placement patterns

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.12 or higher |
| Node.js | Latest LTS version |
| GPU | Recommended for high-frequency batch processing (CPU compatible) |
| API Key | Google Gemini API Key (see [Configuration](#configuration)) |
| OS | macOS, Linux, or Windows 10+ |

---

## Installation

### 1. Repository Structure

Clone the repository and verify the following file structure before proceeding:

```
MineSearcherProject/
├── backend/
│   ├── main.py                  # FastAPI application entry point
│   ├── hazard_model.h5          # Pre-trained CNN weights (required)
│   ├── agent.py                 # Neuro-Symbolic agent logic
│   ├── solver.py                # CSP Constraint Satisfaction Solver
│   ├── xai_engine.py            # Gemma 2 XAI integration
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   ├── App.jsx
    │   └── main.jsx
    ├── index.html
    ├── vite.config.js
    └── package.json
```

> ⚠️ **Critical:** The `hazard_model.h5` file must be present in `backend/` before launching the API. This file contains the pre-trained CNN weights and is required for all simulation and inference endpoints.

---

### 2. Backend Setup

Navigate to the `backend/` directory and create a Python virtual environment:

```bash
cd backend
python -m venv venv

# Activate the virtual environment
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install numpy tensorflow fastapi uvicorn google-generativeai pydantic
pip install fastapi uvicorn google-generativeai pydantic
```

> **Note for GPU Users:** To enable GPU acceleration for batch processing, install the GPU-compatible build of TensorFlow:
> ```bash
> pip install tensorflow[and-cuda]   # Linux with CUDA
> ```

---

### 3. Frontend Setup

Navigate to the `frontend/` directory and install Node.js dependencies:

```bash
cd ../frontend
npm install
```

---

## Running the Platform

The platform requires **two concurrent terminal sessions**.

### Terminal 1 — Backend API

```bash
cd backend
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

uvicorn main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`  
Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`

### Terminal 2 — Frontend Dashboard

```bash
cd frontend
npm run dev
```

The dashboard will be available at: `http://localhost:5173` (default Vite port)

---

## Research Capabilities

MineSearcher is designed to support the following research workflows:

| Workflow | Description |
|----------|-------------|
| **Single Mission Analysis** | Run one simulation and receive a full XAI post-mortem report |
| **Batch Benchmarking** | Execute up to 500 simulations, with automatic complexity classification |
| **Pattern Mining** | Identify grid configurations that maximize agent uncertainty |
| **Adversarial Research** | Export `Hard` battlefield patterns for adversarial training experiments |
| **Architecture Ablation** | Disable the CSP solver to benchmark the CNN-only baseline |

---

## Configuration

### Google Gemini API Key

> ⚠️ **Research Build Notice:** For this research prototype, the API key is hardcoded directly in `main.py`. This approach is intentional for a local, closed research environment but is **not suitable for production or public deployment.**

To configure the API key, locate the following line in `backend/main.py` and replace the placeholder:

```python
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

Obtain a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## Project Status

| Component | Status |
|-----------|--------|
| Neuro-Symbolic Agent (CNN + CSP) | ✅ Stable |
| XAI Engine (Gemma 2 27B) | ✅ Stable |
| Batch Benchmarking (up to 500) | ✅ Stable |
| Interactive Heatmap Visualization | ✅ Stable |
| Battlefield Optimization Engine | ✅ Stable |
| GPU Batch Acceleration | 🔧 Experimental |

---

*MineSearcher v1.0 — Rutgers University, Master's Research Build*