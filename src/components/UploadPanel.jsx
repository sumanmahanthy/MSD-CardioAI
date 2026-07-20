import React, { useState, useCallback, useRef } from "react";
import API from "../services/api";

/* ── Pipeline Stage Definitions ────────────────────────────────────────── */
const STAGES = [
    { label: "Loading & validating image",        pct: 10,  step: 0 },
    { label: "CLAHE quality enhancement",          pct: 25,  step: 1 },
    { label: "Anatomy-aware lung masking (Otsu)",  pct: 42,  step: 2 },
    { label: "ConvNeXt-V2-Tiny forward pass",      pct: 60,  step: 3 },
    { label: "3-Way disease classification",       pct: 76,  step: 4 },
    { label: "Grad-CAM heatmap generation",        pct: 90,  step: 5 },
    { label: "Preparing clinical report",          pct: 98,  step: 6 },
];

const PIPELINE_STEPS = [
    { key: "CLAHE",         icon: "🔬" },
    { key: "Lung Mask",     icon: "🫁" },
    { key: "ConvNeXt",      icon: "🤖" },
    { key: "3-Way Class",   icon: "🫀" },
    { key: "Grad-CAM",      icon: "🔥" },
    { key: "Report",        icon: "📋" },
];

/* ── X-ray Preview with Scanner ────────────────────────────────────────── */
function ImagePreview({ image }) {
    return (
        <div style={{
            marginTop: "16px",
            borderRadius: "14px",
            overflow: "hidden",
            background: "linear-gradient(145deg, #050d1c 0%, #0c1628 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "220px",
            position: "relative",
            border: "1px solid rgba(99,102,241,0.18)",
            boxShadow: "0 8px 32px rgba(0,0,0,0.4), inset 0 0 40px rgba(99,102,241,0.03)",
        }}>
            {/* Corner scanner brackets */}
            {[["top","left"],["top","right"],["bottom","left"],["bottom","right"]].map(([v, h]) => (
                <div key={`${v}-${h}`} style={{
                    position: "absolute",
                    [v]: "10px", [h]: "10px",
                    width: "22px", height: "22px",
                    borderTop:    v === "top"    ? "2px solid rgba(99,102,241,0.8)" : "none",
                    borderBottom: v === "bottom" ? "2px solid rgba(99,102,241,0.8)" : "none",
                    borderLeft:   h === "left"   ? "2px solid rgba(99,102,241,0.8)" : "none",
                    borderRight:  h === "right"  ? "2px solid rgba(99,102,241,0.8)" : "none",
                    zIndex: 3,
                    boxShadow: "0 0 8px rgba(99,102,241,0.4)",
                }} />
            ))}

            {/* Scan line animation */}
            <div style={{
                position: "absolute",
                left: 0, right: 0, top: 0,
                height: "2px",
                background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.8), rgba(34,211,238,0.6), transparent)",
                boxShadow: "0 0 12px rgba(99,102,241,0.6)",
                animation: "scanMove 3s ease-in-out infinite",
                zIndex: 3,
            }} />

            <img
                src={image}
                alt="Uploaded X-ray"
                style={{
                    width: "100%",
                    maxHeight: "270px",
                    objectFit: "contain",
                    display: "block",
                    opacity: 0.92,
                    position: "relative",
                    zIndex: 2,
                }}
            />
        </div>
    );
}

/* ── AI Loading Overlay ─────────────────────────────────────────────────── */
function LoadingOverlay({ stage, progress, stageStep }) {
    return (
        <div style={{
            marginTop: "16px",
            padding: "22px",
            borderRadius: "16px",
            background: "linear-gradient(135deg, rgba(99,102,241,0.06), rgba(34,211,238,0.03))",
            border: "1px solid rgba(99,102,241,0.18)",
            animation: "scaleIn 0.3s ease both",
        }}>
            {/* Header row */}
            <div style={{ display: "flex", alignItems: "center", gap: "14px", marginBottom: "16px" }}>
                <div style={{
                    width: "48px", height: "48px",
                    borderRadius: "13px",
                    background: "rgba(99,102,241,0.1)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    animation: "pulse-ring 1.5s ease-in-out infinite",
                    flexShrink: 0,
                    border: "1px solid rgba(99,102,241,0.2)",
                }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
                        fill="none" stroke="#818cf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="2 12 5 12 7 6 10 18 13 9 15 15 17 12 22 12"/>
                    </svg>
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 700, fontSize: "13px", color: "#818cf8", marginBottom: "3px" }}>
                        {stage || "Initialising AI pipeline..."}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--text-faint)" }}>
                        CardioAI · ConvNeXt-V2-Tiny Student KD · Triple-Teacher Ensemble
                    </div>
                </div>
                <div style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "20px", fontWeight: 700,
                    color: "#818cf8",
                    minWidth: "48px", textAlign: "right",
                }}>
                    {progress}%
                </div>
            </div>

            {/* Progress bar */}
            <div className="ai-progress-bar" style={{ marginBottom: "16px" }}>
                <div className="ai-progress-fill" style={{ width: `${progress}%` }} />
            </div>

            {/* Pipeline step pills */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {PIPELINE_STEPS.map((step, i) => {
                    const isDone   = stageStep > i;
                    const isActive = stageStep === i;
                    return (
                        <span
                            key={step.key}
                            className={`pipeline-step ${isActive ? "active" : isDone ? "done" : "idle"}`}
                        >
                            {isDone ? "✓" : step.icon} {step.key}
                        </span>
                    );
                })}
            </div>
        </div>
    );
}

/* ── Main UploadPanel ───────────────────────────────────────────────────── */
function UploadPanel({ setPrediction }) {
    const [image, setImage]           = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);   // actual File object
    const [loading, setLoading]       = useState(false);
    const [progress, setProgress]     = useState(0);
    const [stage, setStage]           = useState("");
    const [stageStep, setStageStep]   = useState(-1);
    const [errorMessage, setError]    = useState("");
    const [isDragOver, setIsDragOver] = useState(false);
    const fileRef                     = useRef(null);

    const processFile = useCallback((file) => {
        if (!file) return;
        setSelectedFile(file);                          // store real File object
        setImage(URL.createObjectURL(file));
        setPrediction(null);
        setError("");
    }, [setPrediction]);

    const handleFileChange = (e) => processFile(e.target.files[0]);
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragOver(false);
        processFile(e.dataTransfer.files[0]);
    };

    const analyzeImage = async () => {
        // Works for both dialog-pick AND drag-and-drop
        const file = selectedFile || fileRef.current?.files[0];
        if (!file) return;

        setLoading(true);
        setProgress(0);
        setStageStep(-1);
        setPrediction(null);
        setError("");

        let stageIdx = 0;
        const tick = setInterval(() => {
            if (stageIdx < STAGES.length) {
                setStage(STAGES[stageIdx].label);
                setProgress(STAGES[stageIdx].pct);
                setStageStep(STAGES[stageIdx].step);
                stageIdx++;
            }
        }, 440);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await API.post("/predict", formData);
            clearInterval(tick);
            setProgress(100);
            setStage("Analysis complete!");
            setStageStep(PIPELINE_STEPS.length);

            // Persist to notification store
            const entry = {
                id:         Date.now(),
                label:      response.data.label,
                confidence: response.data.confidence,
                time:       new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                read:       false,
            };
            const prev    = JSON.parse(localStorage.getItem("cardio_notifications") || "[]");
            localStorage.setItem("cardio_notifications", JSON.stringify([entry, ...prev].slice(0, 10)));
            window.dispatchEvent(new Event("cardio_notify"));

            setTimeout(() => {
                setPrediction(response.data);
                setLoading(false);
                setProgress(0);
                setStage("");
                setStageStep(-1);
            }, 750);

        } catch (err) {
            clearInterval(tick);
            const data   = err.response?.data;
            const status = err.response?.status;

            if (status === 422 && data?.error === "invalid_image") {
                setError(`${data.message}\n\n${data.hint || ""}`);
                setStage("Not a valid chest X-ray. Please retry.");
            } else {
                const msg = data?.message || data?.detail || err.message || "Unknown error";
                setError(msg);
                setStage("Analysis failed. Please retry.");
            }

            setProgress(0);
            setStageStep(-1);
            setTimeout(() => { setLoading(false); setStage(""); }, 2500);
        }
    };

    const clearAll = () => {
        setImage(null);
        setSelectedFile(null);
        setPrediction(null);
        setError("");
        setStageStep(-1);
        if (fileRef.current) fileRef.current.value = "";
    };

    return (
        <div className="card p-4 border-0 animate-fade-up">

            {/* ── Header ── */}
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "20px" }}>
                <div className="stat-icon icon-indigo" style={{ width: "42px", height: "42px", borderRadius: "11px" }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                </div>
                <div>
                    <h6 style={{ margin: 0, fontWeight: 800, color: "var(--text-main)", fontSize: "15px" }}>
                        Upload Chest X-ray
                    </h6>
                    <p style={{ margin: 0, fontSize: "11px", color: "var(--text-faint)" }}>
                        PA view · DICOM or JPG/PNG · Max 10MB
                    </p>
                </div>
            </div>

            {/* ── Drop Zone ── */}
            <div
                className={`upload-dropzone${isDragOver ? " drag-over" : ""}`}
                onClick={() => !loading && fileRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                style={{ cursor: loading ? "not-allowed" : "pointer", opacity: loading ? 0.5 : 1 }}
            >
                {/* Glowing upload icon */}
                <div style={{
                    width: "56px", height: "56px", margin: "0 auto 14px",
                    borderRadius: "15px",
                    background: "rgba(99,102,241,0.1)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    border: "1px solid rgba(99,102,241,0.2)",
                    boxShadow: isDragOver ? "0 0 20px rgba(99,102,241,0.4)" : "0 0 12px rgba(99,102,241,0.12)",
                    transition: "var(--transition-base)",
                }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
                        fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    </svg>
                </div>
                <p style={{ margin: 0, fontWeight: 600, fontSize: "13.5px", color: "var(--text-secondary)" }}>
                    {isDragOver ? "Drop your X-ray here" : "Drag & drop or click to browse"}
                </p>
                <p style={{ margin: "5px 0 0", fontSize: "11px", color: "var(--text-faint)" }}>
                    Supports PNG, JPG, JPEG · DICOM converted automatically
                </p>

                {/* Format pills */}
                <div style={{ display: "flex", gap: "6px", justifyContent: "center", marginTop: "14px" }}>
                    {["PNG", "JPG", "JPEG", "DICOM"].map(fmt => (
                        <span key={fmt} style={{
                            fontSize: "10px", padding: "3px 9px",
                            borderRadius: "99px", fontWeight: 700,
                            background: "rgba(99,102,241,0.08)",
                            border: "1px solid rgba(99,102,241,0.15)",
                            color: "var(--text-faint)",
                        }}>{fmt}</span>
                    ))}
                </div>
            </div>

            <input
                ref={fileRef}
                type="file"
                id="fileInput"
                accept="image/*"
                onChange={handleFileChange}
                disabled={loading}
                style={{ display: "none" }}
            />

            {/* ── Preview ── */}
            {image && !loading && <ImagePreview image={image} />}

            {/* ── Loading State ── */}
            {loading && <LoadingOverlay stage={stage} progress={progress} stageStep={stageStep} />}

            {/* ── Action Buttons ── */}
            <div style={{ display: "flex", gap: "10px", marginTop: "18px" }}>
                <button
                    className="btn btn-primary fw-bold"
                    style={{ flex: 2 }}
                    onClick={analyzeImage}
                    disabled={!image || loading}
                >
                    {loading ? (
                        <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
                            <span className="spinner-border spinner-border-sm" role="status" />
                            Analysing...
                        </span>
                    ) : (
                        <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
                                fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                            </svg>
                            Analyze X-ray
                        </span>
                    )}
                </button>

                <button
                    className="btn btn-outline-danger fw-bold"
                    style={{ flex: 1 }}
                    onClick={clearAll}
                    disabled={loading || (!image && !errorMessage)}
                >
                    Clear
                </button>
            </div>

            {/* ── Error ── */}
            {errorMessage && (
                <div style={{
                    marginTop: "14px",
                    padding: "14px 16px",
                    borderRadius: "12px",
                    background: "rgba(239,68,68,0.06)",
                    border: "1px solid rgba(239,68,68,0.2)",
                    animation: "fadeIn 0.3s ease",
                }}>
                    <div style={{ fontWeight: 700, fontSize: "13px", color: "#f87171", marginBottom: "5px", display: "flex", alignItems: "center", gap: "6px" }}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
                            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                        </svg>
                        Upload Failed
                    </div>
                    <div style={{ fontSize: "12px", color: "var(--text-muted)", whiteSpace: "pre-line" }}>
                        {errorMessage}
                    </div>
                </div>
            )}
        </div>
    );
}

export default UploadPanel;
