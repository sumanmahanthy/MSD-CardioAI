import React, { useState } from "react";
import UploadPanel from "../components/UploadPanel";
import PredictionResult from "../components/PredictionResult";
import ProbabilityChart from "../components/ProbabilityChart";
import HeatmapViewer from "../components/HeatmapViewer";
import MultiDiseaseChart from "../components/MultiDiseaseChart";

function Upload() {
    const [prediction, setPrediction] = useState(null);

    return (
        <div className="upload-page">

            {/* ── Page Hero ── */}
            <div style={{
                padding: "22px 28px",
                borderRadius: "20px",
                background: "linear-gradient(135deg, rgba(99,102,241,0.07) 0%, rgba(34,211,238,0.04) 50%, transparent 100%)",
                border: "1px solid var(--border-color)",
                marginBottom: "24px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: "16px",
                position: "relative",
                overflow: "hidden",
            }}>
                {/* Background grid */}
                <div style={{
                    position: "absolute", inset: 0,
                    backgroundImage: "radial-gradient(rgba(99,102,241,0.04) 1px, transparent 1px)",
                    backgroundSize: "28px 28px",
                    pointerEvents: "none",
                }} />

                <div style={{ position: "relative", zIndex: 1 }}>
                    <div style={{
                        display: "inline-flex", alignItems: "center", gap: "7px",
                        padding: "3px 12px",
                        background: "rgba(99,102,241,0.1)",
                        border: "1px solid rgba(99,102,241,0.22)",
                        borderRadius: "99px",
                        fontSize: "10px", fontWeight: 700,
                        color: "var(--secondary)",
                        textTransform: "uppercase", letterSpacing: "0.8px",
                        marginBottom: "10px",
                    }}>
                        <span className="live-dot" />
                        AI Pipeline Ready
                    </div>
                    <h3 className="page-title" style={{ marginBottom: "5px" }}>Analyse Chest X-ray</h3>
                    <p style={{ margin: 0, fontSize: "13px", color: "var(--text-muted)" }}>
                        Frontal PA view · ConvNeXt-V2-Tiny · Results in under 3 seconds
                    </p>
                </div>

                {/* Right: pipeline stage chips */}
                <div style={{
                    display: "flex", gap: "6px", flexWrap: "wrap",
                    position: "relative", zIndex: 1,
                }}>
                    {[
                        { icon: "🔬", label: "CLAHE" },
                        { icon: "🫁", label: "Lung Mask" },
                        { icon: "🤖", label: "ConvNeXt" },
                        { icon: "🫀", label: "3-Way" },
                        { icon: "🔥", label: "Grad-CAM++" },
                    ].map(s => (
                        <span key={s.label} style={{
                            padding: "5px 11px",
                            borderRadius: "99px",
                            background: "rgba(99,102,241,0.07)",
                            border: "1px solid var(--border-color)",
                            fontSize: "11px", fontWeight: 600,
                            color: "var(--text-faint)",
                        }}>
                            {s.icon} {s.label}
                        </span>
                    ))}
                </div>
            </div>

            <div className="row g-4">
                {/* ── Upload Panel ── */}
                <div className={prediction ? "col-lg-5" : "col-lg-7 mx-auto"}>
                    <UploadPanel setPrediction={setPrediction} />
                </div>

                {/* ── Results ── */}
                {prediction && (
                    <div className="col-lg-7">

                        {/* Diagnosis card */}
                        <div style={{ marginBottom: "16px" }}>
                            <PredictionResult result={prediction} />
                        </div>

                        {/* Heatmap + Probability side-by-side */}
                        <div className="row g-4" style={{ marginBottom: "16px" }}>
                            <div className="col-md-6">
                                <HeatmapViewer heatmap={prediction.heatmap_url} label={prediction.label} />
                            </div>
                            <div className="col-md-6">
                                <ProbabilityChart prediction={prediction.probabilities} />
                            </div>
                        </div>

                        {/* Multi-disease breakdown (conditional) */}
                        {prediction.multi_disease && Object.keys(prediction.multi_disease).length > 0 && (
                            <MultiDiseaseChart data={prediction.multi_disease} />
                        )}

                    </div>
                )}
            </div>
        </div>
    );
}

export default Upload;