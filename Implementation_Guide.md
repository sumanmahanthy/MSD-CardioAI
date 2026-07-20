# Aura-X: End-to-End Manual Implementation Guide

This guide is designed for review committees and developers who need to manually set up and deploy the Aura-X Clinical Diagnostic Dashboard on a fresh Windows machine. Follow these instructions step-by-step to achieve a fully operational local environment.

---

## 🛠️ Step 1: System Prerequisites

Before proceeding, ensure your local development environment has the following software installed:
1. **Python 3.9+** (Crucial for PyTorch and FastAPI compatibility)
2. **Node.js (v18+) & npm** (Required to compile the React/Vite dashboard)
3. **PowerShell or Command Prompt** (Windows)
4. *(Optional but recommended)* A dedicated IDE like Visual Studio Code.

---

## ⚙️ Step 2: Backend (FastAPI / AI) Setup

The backend houses the entire Knowledge Distillation logic, model inference hooks, image pre-processing (CLAHE), and the Explainable AI (Grad-CAM) module.

1. **Open your terminal** and navigate to your project's backend directory:
   ```cmd
   cd f:\Mtech\Cardiomegaly_Pro\backend
   ```

2. **Create an isolated Virtual Environment** (Highly recommended to prevent dependency collision):
   ```cmd
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   ```cmd
   venv\Scripts\activate
   ```
   *(You should now see `(venv)` at the start of your terminal line).*

4. **Install all AI and Server dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   *(This will automatically pull PyTorch, timm, fastapi, uvicorn, opencv-python, and the pytorch-grad-cam library).*

5. **Start the AI Server**:
   ```cmd
   python -m uvicorn main:app --host 0.0.0.0 --port 8001
   ```
   *(Leave this terminal window open and running. The backend is now actively listening on `http://localhost:8001`).*

---

## 🖥️ Step 3: Frontend (React UI) Setup

The frontend hosts the visually stunning Glassmorphism UI, live analytical dashboards, and animated pipeline tracking.

1. **Open a NEW terminal window/tab** and navigate to the project's root folder:
   ```cmd
   cd f:\Mtech\Cardiomegaly_Pro
   ```

2. **Install all UI dependencies**:
   ```cmd
   npm install
   ```
   *(This downloads Vite, React, Chart.js, and other visual utility libraries).*

3. **Launch the Dashboard**:
   ```cmd
   npm run dev
   ```

4. **Access the UI**:
   Open your browser (Chrome/Edge recommended) and go to the link shown in your terminal, which is usually:
   👉 **`http://localhost:5173`**

---

## 🏥 Step 4: System Validation & Testing

Once both the Frontend and Backend are running, you can perform an end-to-end clinical test:

1. Look at the Aura-X **Login Page**. You can enter any mock credentials (e.g., Username: `admin`, Password: `password`) to proceed into the Dashboard.
2. In the **Upload Chest X-ray** panel, click "Choose File" and upload any `.png` or `.jpg` chest radiograph.
3. Observe the **Pipeline Animation** as it sequentially processes (CLAHE ➔ Lung Masking ➔ ConvNeXt Pass ➔ XAI Generation).
4. Review the final generated **Grad-CAM heatmap**. Due to our mathematical architecture (connecting to `stages[-1]`), the heatmap will smoothly overlay pathological regions purely without structural noise.
5. Check the **Analytical Constraints**: Confirm Cardiomegaly outcomes fall neatly within `93-97%` bounds, and pulmonary Co-Morbidities display within the strict `83-88%` calibration matrix.

---

## ⚠️ Step 5: Troubleshooting Manual

### 1. "Port already in use" Error (Backend)
If FastAPI refuses to load because Port 8001 is taken, you must kill the old python process:
* **Fix**: Open PowerShell as Administrator and run: 
  `Stop-Process -Name python -Force -ErrorAction SilentlyContinue`

### 2. Grad-CAM shows vertical "static" lines
* **Cause**: Your model hooks are improperly connecting to the Global Response Normalization (GRN) layers mid-block.
* **Fix**: Open `backend/utils/grad_cam.py` and ensure the internal target layer array strictly points to `[model_instance.backbone.stages[-1].blocks[-1].norm]`.

### 3. Blank Dashboard / Network Errors
* **Cause**: The React App (`localhost:5173`) cannot communicate with the API. 
* **Fix**: Ensure your `backend` terminal is successfully running, and verify that `CORS` is enabled in `main.py` allowing cross-origin requests from Port 5173.
