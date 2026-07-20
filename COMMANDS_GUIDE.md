# CardioAI Pro V6 — Start / Stop Command Guide

> **OS:** Windows · **Shell:** PowerShell · **Project root:** `F:\Mtech\Cardiomegaly_Pro`

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [First-time Setup](#2-first-time-setup)
3. [Start the Backend (FastAPI)](#3-start-the-backend-fastapi)
4. [Start the Frontend (React + Vite)](#4-start-the-frontend-react--vite)
5. [Verify Both Are Running](#5-verify-both-are-running)
6. [Stop the Backend](#6-stop-the-backend)
7. [Stop the Frontend](#7-stop-the-frontend)
8. [Stop Everything at Once](#8-stop-everything-at-once)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

Make sure the following are installed before running anything:

| Tool | Check command | Required version |
|------|--------------|-----------------|
| Python | `python --version` | >= 3.10 |
| pip | `pip --version` | >= 23 |
| Node.js | `node --version` | >= 18 |
| npm | `npm --version` | >= 9 |

> IMPORTANT: The `best_student.pth` model weights file must exist at:
> `F:\Mtech\Cardiomegaly_Pro\backend\models\best_student.pth`
> Without it, the backend runs in **demo simulation mode** (random predictions).

---

## 2. First-time Setup

### Backend Python dependencies
```powershell
# Open PowerShell and navigate to backend folder
cd F:\Mtech\Cardiomegaly_Pro\backend

# (Recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all required packages
pip install -r requirements.txt
```

### Frontend Node.js dependencies
```powershell
# Navigate to project root (where package.json lives)
cd F:\Mtech\Cardiomegaly_Pro

# Install Node packages (only needed once)
npm install
```

> NOTE: You only need to run setup steps once. After that, skip directly to Section 3.

---

## 3. Start the Backend (FastAPI)

Open **Terminal / PowerShell Window 1** and run:

```powershell
# 1. Go to backend directory
cd F:\Mtech\Cardiomegaly_Pro\backend

# 2. Activate virtual environment (if you created one)
.\venv\Scripts\Activate.ps1

# 3. Start FastAPI server on port 8001
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Expected output when running correctly:**
```
[ModelPipeline] Device: cpu
[ModelPipeline] Loading ConvNeXt-V2-Tiny student from models/best_student.pth...
[ModelPipeline] OK Student model loaded (28.6M params).
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> TIP: The `--reload` flag auto-restarts the server when you edit Python files.
> Remove it in production: `uvicorn main:app --host 0.0.0.0 --port 8001`

---

## 4. Start the Frontend (React + Vite)

Open **Terminal / PowerShell Window 2** (keep Window 1 running) and run:

```powershell
# 1. Go to project root
cd F:\Mtech\Cardiomegaly_Pro

# 2. Start the Vite development server
npm run dev
```

**Expected output when running correctly:**
```
  VITE v5.x.x  ready in 432 ms

  Local:   http://localhost:5173/
  Network: http://10.x.x.x:5173/
  press h + enter to show help
```

Open your browser and navigate to: **http://localhost:5173**

---

## 5. Verify Both Are Running

### Check backend health
```powershell
# In any terminal window
Invoke-WebRequest -Uri http://localhost:8001 -UseBasicParsing | Select-Object -ExpandProperty Content
```
Expected response:
```json
{"status":"CardioAI Pro V6 backend running","pipeline":"Dual-Teacher Ensemble -> ConvNeXt-V2-Tiny Student",...}
```

### Check frontend
Open browser → `http://localhost:5173` — you should see the CardioAI Pro login page.

### Check which ports are in use
```powershell
# See what is listening on port 8001 (backend)
netstat -ano | findstr :8001

# See what is listening on port 5173 (frontend)
netstat -ano | findstr :5173
```

---

## 6. Stop the Backend

### Option A — Keyboard shortcut (recommended)
In **Terminal Window 1** (where uvicorn is running):
```
Press  Ctrl + C
```
You will see:
```
INFO:     Shutting down
INFO:     Finished server process
```

### Option B — Kill by process ID
```powershell
# Find the PID listening on port 8001
netstat -ano | findstr :8001

# Kill that process (replace 12345 with actual PID)
taskkill /PID 12345 /F
```

### Option C — Kill all uvicorn / python processes
```powershell
# WARNING: This kills ALL python processes on your machine
Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
```

---

## 7. Stop the Frontend

### Option A — Keyboard shortcut (recommended)
In **Terminal Window 2** (where Vite is running):
```
Press  Ctrl + C
```
The Vite server stops immediately.

### Option B — Kill by process ID
```powershell
# Find PID on port 5173
netstat -ano | findstr :5173

# Kill it (replace 67890 with actual PID)
taskkill /PID 67890 /F
```

### Option C — Kill all node processes
```powershell
# WARNING: This kills ALL node.js processes on your machine
Stop-Process -Name "node" -Force -ErrorAction SilentlyContinue
```

---

## 8. Stop Everything at Once

```powershell
# Stop backend (uvicorn/python) and frontend (node/vite) in one go
Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "node"   -Force -ErrorAction SilentlyContinue
Write-Host "All CardioAI Pro services stopped." -ForegroundColor Green
```

> CAUTION: The above command stops ALL Python and Node processes on your machine,
> not just CardioAI. Use it only if no other Python/Node apps are running.

---

## 9. Troubleshooting

### ERROR: uvicorn not found
```powershell
# Make sure your venv is activated, then install uvicorn
.\venv\Scripts\Activate.ps1
pip install uvicorn
```

### ERROR: Address already in use — port 8001
```powershell
# Find the PID and kill it
$PID8001 = (netstat -ano | findstr ":8001 " | ForEach-Object { ($_ -split "\s+")[-1] } | Select-Object -First 1)
if ($PID8001) { taskkill /PID $PID8001 /F }
# Then restart the backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### ERROR: Address already in use — port 5173
```powershell
# Start Vite on a different port
npm run dev -- --port 5174
```
Then update `backend/main.py` CORS allowed origins to include `http://localhost:5174`.

### ERROR: Backend returns 500 — model not loading
- Check that `backend\models\best_student.pth` exists.
- If missing, the backend falls back to **demo mode** (random predictions).
- Rerun `cardioai_v6_student_ensemble.py` on Kaggle to regenerate the checkpoint.

### ERROR: Frontend shows blank page / network error
- Confirm the backend is running: `Invoke-WebRequest http://localhost:8001`
- Check browser console (F12) for CORS errors.
- Make sure `src/services/api.js` has `baseURL: "http://localhost:8001"`.

### ERROR: ModuleNotFoundError: No module named 'timm'
```powershell
cd F:\Mtech\Cardiomegaly_Pro\backend
.\venv\Scripts\Activate.ps1
pip install timm
```

### ERROR: ImportError: pytorch_grad_cam not installed
Grad-CAM will fall back to a demo heatmap — this is safe and will not crash.
To enable real Grad-CAM:
```powershell
pip install pytorch-grad-cam
```

### ERROR: ExecutionPolicy prevents running venv activation script
```powershell
# Run this once as Administrator to allow local scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| Start backend | `cd backend ; uvicorn main:app --host 0.0.0.0 --port 8001 --reload` |
| Start frontend | `cd F:\Mtech\Cardiomegaly_Pro ; npm run dev` |
| Stop backend | `Ctrl+C` in backend terminal |
| Stop frontend | `Ctrl+C` in frontend terminal |
| Stop all (force) | `Stop-Process -Name python,node -Force` |
| Check backend health | `Invoke-WebRequest http://localhost:8001` |
| View API docs (Swagger) | Open `http://localhost:8001/docs` in browser |
| View frontend app | Open `http://localhost:5173` in browser |
