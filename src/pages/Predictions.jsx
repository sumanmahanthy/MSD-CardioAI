import React from "react";
import { useNavigate } from "react-router-dom";

/**
 * Predictions page — redirects the user to Upload page.
 * The actual live prediction results are shown on the Upload page
 * immediately after analysis. This page acts as a shortcut entry point.
 */
function Predictions() {
  const navigate = useNavigate();

  return (
    <div className="predictions-page">

      <div className="d-flex align-items-center justify-content-between mb-4">
        <div>
          <h3 className="mb-1 fw-bold text-main">Prediction Results</h3>
          <p className="text-muted mb-0" style={{ fontSize: "14px" }}>
            Upload and analyse a chest X-ray to see live AI results with Grad-CAM.
          </p>
        </div>
      </div>

      {/* Info card */}
      <div style={{
        background: "var(--card-bg)",
        border: "1px solid var(--border-color, #e2e8f0)",
        borderRadius: "20px",
        padding: "60px 40px",
        textAlign: "center",
        boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
      }}>
        {/* Animated ECG icon */}
        <div style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: "90px",
          height: "90px",
          borderRadius: "24px",
          background: "linear-gradient(135deg, rgba(37,99,235,0.1), rgba(99,102,241,0.1))",
          marginBottom: "24px",
        }}>
          <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24"
            fill="none" stroke="url(#pg)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <defs>
              <linearGradient id="pg" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#2563eb" />
                <stop offset="100%" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
            <polyline points="2 12 5 12 7 6 10 18 13 9 15 15 17 12 22 12" />
          </svg>
        </div>

        <h4 style={{ fontWeight: "700", color: "var(--text-main, #1e293b)", marginBottom: "12px" }}>
          No Active Analysis
        </h4>
        <p style={{ color: "#94a3b8", fontSize: "15px", maxWidth: "400px", margin: "0 auto 28px" }}>
          Upload a chest X-ray image and the AI pipeline will produce live results
          including Grad-CAM heatmaps, confidence scores, and multi-disease classification.
        </p>

        <button
          onClick={() => navigate("/upload")}
          style={{
            padding: "12px 32px",
            borderRadius: "12px",
            border: "none",
            background: "linear-gradient(135deg, #2563eb, #7c3aed)",
            color: "#fff",
            fontWeight: "700",
            fontSize: "15px",
            cursor: "pointer",
            boxShadow: "0 4px 20px rgba(37,99,235,0.35)",
            transition: "transform 0.2s, box-shadow 0.2s",
            marginRight: "12px",
          }}
          onMouseEnter={e => { e.target.style.transform = "translateY(-2px)"; e.target.style.boxShadow = "0 8px 28px rgba(37,99,235,0.4)"; }}
          onMouseLeave={e => { e.target.style.transform = "translateY(0)"; e.target.style.boxShadow = "0 4px 20px rgba(37,99,235,0.35)"; }}
        >
          🩻 Upload X-ray
        </button>
        <button
          onClick={() => navigate("/history")}
          style={{
            padding: "12px 28px",
            borderRadius: "12px",
            border: "1px solid var(--border-color, #e2e8f0)",
            background: "transparent",
            color: "var(--text-muted, #64748b)",
            fontWeight: "600",
            fontSize: "15px",
            cursor: "pointer",
            transition: "all 0.2s",
          }}
          onMouseEnter={e => { e.target.style.background = "var(--border-color, #f1f5f9)"; }}
          onMouseLeave={e => { e.target.style.background = "transparent"; }}
        >
          📁 View History
        </button>
      </div>

    </div>
  );
}

export default Predictions;