import React from "react";
import PredictionHistory from "../components/PredictionHistory";

function History() {
    return (
        <div className="history-page">

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
                        Patient Database
                    </div>
                    <h3 className="page-title" style={{ marginBottom: "5px" }}>Prediction History</h3>
                    <p style={{ margin: 0, fontSize: "13px", color: "var(--text-muted)" }}>
                        Full log of all analysed chest X-rays with AI results and Grad-CAM heatmaps.
                    </p>
                </div>
            </div>

            {/* ── History Table ── */}
            <div className="row">
                <div className="col-12">
                    <div className="card p-4 border-0" style={{ borderRadius: "20px", minHeight: "600px" }}>
                        <PredictionHistory />
                    </div>
                </div>
            </div>

        </div>
    );
}

export default History;