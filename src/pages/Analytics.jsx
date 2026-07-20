import React from "react";

import AnalyticsPanel from "../components/AnalyticsPanel";
import ModelComparisonChart from "../components/ModelComparisonChart";
import PredictionDistributionChart from "../components/PredictionDistributionChart";
import ConfusionMatrix from "../components/ConfusionMatrix";
import RocCurve from "../components/RocCurve";
import TrainingHistory from "../components/TrainingHistory";
import ModelArchitecture from "../components/ModelArchitecture";

function Analytics() {
    return (
        <div className="analytics-page">

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
                        Model Analytics
                    </div>
                    <h3 className="page-title" style={{ marginBottom: "5px" }}>Performance & Architecture</h3>
                    <p style={{ margin: 0, fontSize: "13px", color: "var(--text-muted)" }}>
                        Deep-dive metrics for the Triple-Teacher Knowledge Distillation pipeline.
                    </p>
                </div>
            </div>

            {/* Top Statistics */}
            <div style={{ marginBottom: "24px" }}>
                <AnalyticsPanel />
            </div>

            {/* Proposed Architecture — Triple-Teacher KD Pipeline */}
            <div style={{ marginBottom: "24px" }}>
                <ModelArchitecture />
            </div>

            {/* Model Performance Comparison */}
            <div style={{ marginBottom: "24px" }}>
                <ModelComparisonChart />
            </div>

            {/* Prediction Distribution + Confusion Matrix */}
            <div className="row g-4" style={{ marginBottom: "24px" }}>
                <div className="col-lg-6">
                    <PredictionDistributionChart />
                </div>
                <div className="col-lg-6">
                    <ConfusionMatrix />
                </div>
            </div>

            {/* ROC Curve */}
            <div style={{ marginBottom: "24px" }}>
                <RocCurve />
            </div>

            {/* Training History — Macro-AUC per epoch */}
            <div style={{ marginBottom: "24px" }}>
                <TrainingHistory />
            </div>

        </div>
    );
}

export default Analytics;