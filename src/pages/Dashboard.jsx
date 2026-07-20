import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import API from "../services/api";
import DashboardCards from "../components/DashboardCards";
import PredictionDistributionChart from "../components/PredictionDistributionChart";
import PredictionHistory from "../components/PredictionHistory";
import AnalyticsPanel from "../components/AnalyticsPanel";

/* ── ECG Waveform SVG ── */
function EcgWave() {
    return (
        <svg className="ecg-svg" width="420" height="80" viewBox="0 0 420 80" preserveAspectRatio="none">
            <path
                className="ecg-path"
                d="M0,50 L50,50 L60,50 L65,10 L70,70 L75,50 L90,50 L95,30 L100,70 L105,50 L120,50 L125,45 L130,55 L135,50 L160,50 L165,5 L170,75 L175,50 L200,50 L205,35 L210,65 L215,50 L240,50 L245,40 L250,60 L255,50 L280,50 L285,8 L290,72 L295,50 L320,50 L325,30 L330,70 L335,50 L360,50 L365,42 L370,58 L375,50 L420,50"
            />
        </svg>
    );
}

/* ── Compact Mini History for Hero ── */
function HeroHistory() {
    const [history, setHistory] = useState([]);
    useEffect(() => {
        API.get("/prediction-history")
            .then(res => setHistory(res.data.slice(0, 4)))
            .catch(() => {});
    }, []);

    const getStyle = (pred) => {
        if (pred === "Cardiomegaly") return { color: "#f87171", bg: "rgba(239,68,68,0.1)", dot: "#f87171" };
        if (pred === "No Finding")  return { color: "#4ade80", bg: "rgba(34,197,94,0.1)",  dot: "#4ade80" };
        return { color: "#fbbf24", bg: "rgba(245,158,11,0.1)", dot: "#fbbf24" };
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "7px", minWidth: 0 }}>
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                <span style={{ fontSize: "10.5px", fontWeight: 700, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.8px" }}>
                    Recent Analyses
                </span>
                <Link to="/history" style={{ fontSize: "10px", color: "var(--primary)", textDecoration: "none", fontWeight: 700 }}>
                    View all →
                </Link>
            </div>

            {/* Rows */}
            {history.length === 0 ? (
                [1,2,3].map(i => (
                    <div key={i} style={{
                        height: "38px", borderRadius: "10px",
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid var(--border-color)",
                    }} />
                ))
            ) : history.map((entry) => {
                const s = getStyle(entry.prediction);
                return (
                    <div key={entry.id} style={{
                        display: "flex", alignItems: "center", gap: "10px",
                        padding: "7px 11px",
                        borderRadius: "10px",
                        background: s.bg,
                        border: `1px solid ${s.dot}30`,
                    }}>
                        <span style={{
                            width: "7px", height: "7px", borderRadius: "50%",
                            background: s.dot, flexShrink: 0,
                            boxShadow: `0 0 6px ${s.dot}80`,
                        }} />
                        <span style={{
                            fontSize: "12px", fontWeight: 700, color: s.color,
                            flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                        }}>
                            {entry.prediction}
                        </span>
                        <span style={{
                            fontSize: "11px", fontWeight: 700, color: s.color,
                            fontFamily: "'JetBrains Mono', monospace", flexShrink: 0,
                        }}>
                            {(entry.confidence * 100).toFixed(1)}%
                        </span>
                        <span style={{ fontSize: "10px", color: "var(--text-faint)", flexShrink: 0 }}>
                            {entry.date?.split(" ")[1]?.slice(0, 5) || ""}
                        </span>
                    </div>
                );
            })}
        </div>
    );
}

function Dashboard() {
    return (
        <div className="dashboard-page">

            {/* ── Hero Banner ── */}
            <div className="dashboard-hero animate-fade-up">
                <EcgWave />

                {/* Two-column hero layout */}
                <div style={{ position: "relative", zIndex: 1, display: "flex", gap: "32px", alignItems: "flex-start" }}>

                    {/* LEFT — Title + Metrics */}
                    <div style={{ flex: "1 1 0", minWidth: 0 }}>
                        <div className="hero-tag">
                            <span className="live-dot" />
                            Live · CardioAI v6 · ConvNeXt-V2-Tiny
                        </div>
                        <h1 className="hero-title">CardioAI Clinical Intelligence Dashboard</h1>
                        <p className="hero-sub">
                            Triple-Teacher Ensemble → Student Distillation · NIH ChestX-ray14 · 14 Disease Classes
                        </p>
                        <div className="hero-metrics">
                            <div className="hero-metric">
                                <span className="hero-metric-val">93.79<span style={{ fontSize: "14px", fontWeight: 700, color: "var(--secondary)" }}>%</span></span>
                                <span className="hero-metric-lbl">Macro-AUC</span>
                            </div>
                            <div className="hero-divider" />
                            <div className="hero-metric">
                                <span className="hero-metric-val">27.9<span style={{ fontSize: "14px", fontWeight: 700, color: "var(--secondary)" }}>M</span></span>
                                <span className="hero-metric-lbl">Parameters</span>
                            </div>
                            <div className="hero-divider" />
                            <div className="hero-metric">
                                <span className="hero-metric-val">14</span>
                                <span className="hero-metric-lbl">Diseases</span>
                            </div>
                            <div className="hero-divider" />
                            <div className="hero-metric">
                                <span className="hero-metric-val">112<span style={{ fontSize: "14px", fontWeight: 700, color: "var(--secondary)" }}>K</span></span>
                                <span className="hero-metric-lbl">Training Imgs</span>
                            </div>
                            <div className="hero-divider" />
                            <div className="hero-metric">
                                <span className="hero-metric-val" style={{ color: "#4ade80" }}>LIVE</span>
                                <span className="hero-metric-lbl">Backend Status</span>
                            </div>
                        </div>
                    </div>

                    {/* RIGHT — Compact Recent History Panel */}
                    <div style={{
                        flex: "0 0 264px",
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid var(--border-subtle)",
                        borderRadius: "14px",
                        padding: "14px 16px",
                        backdropFilter: "blur(8px)",
                        alignSelf: "stretch",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                    }}>
                        <HeroHistory />
                    </div>
                </div>
            </div>

            {/* ── Section: Clinical KPIs ── */}
            <div className="card-section-label">
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none"
                    stroke="var(--primary)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                Clinical KPIs
            </div>
            <AnalyticsPanel />

            {/* ── Section: Model Benchmarks ── */}
            <div className="card-section-label">
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none"
                    stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                Model Benchmarks
            </div>
            <DashboardCards />

            {/* ── Charts + History Row ── */}
            <div className="row g-4">

                {/* Distribution Donut */}
                <div className="col-lg-5">
                    <div className="card border-0 h-100 animate-fade-up" style={{ animationDelay: "100ms", overflow: "hidden" }}>
                        <div style={{
                            padding: "18px 22px 14px",
                            borderBottom: "1px solid var(--border-subtle)",
                            display: "flex", alignItems: "center", gap: "10px",
                        }}>
                            <div className="stat-icon icon-indigo" style={{ width: "34px", height: "34px", borderRadius: "9px" }}>
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
                                    fill="none" stroke="currentColor" strokeWidth="2"
                                    strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>
                                    <path d="M22 12A10 10 0 0 0 12 2v10z"/>
                                </svg>
                            </div>
                            <div>
                                <h6 style={{ margin: 0, fontWeight: 700, fontSize: "13px", color: "var(--text-main)" }}>
                                    Prediction Distribution
                                </h6>
                                <p style={{ margin: 0, fontSize: "10.5px", color: "var(--text-faint)" }}>
                                    3-way classification breakdown
                                </p>
                            </div>
                        </div>
                        <div style={{ padding: "8px" }}>
                            <PredictionDistributionChart />
                        </div>
                    </div>
                </div>

                {/* Recent Analyses */}
                <div className="col-lg-7">
                    <div className="card border-0 h-100 animate-fade-up" style={{ animationDelay: "180ms", overflow: "hidden" }}>
                        <div style={{
                            padding: "18px 22px 14px",
                            borderBottom: "1px solid var(--border-subtle)",
                            display: "flex", alignItems: "center", justifyContent: "space-between",
                        }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                                <div className="stat-icon icon-green" style={{ width: "34px", height: "34px", borderRadius: "9px" }}>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
                                        fill="none" stroke="currentColor" strokeWidth="2"
                                        strokeLinecap="round" strokeLinejoin="round">
                                        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                                    </svg>
                                </div>
                                <div>
                                    <h6 style={{ margin: 0, fontWeight: 700, fontSize: "13px", color: "var(--text-main)" }}>
                                        Recent AI Analyses
                                    </h6>
                                    <p style={{ margin: 0, fontSize: "10.5px", color: "var(--text-faint)" }}>
                                        Latest X-ray prediction results
                                    </p>
                                </div>
                            </div>
                            <Link to="/history" style={{
                                fontSize: "11.5px", fontWeight: 700,
                                color: "var(--primary)", textDecoration: "none",
                                display: "flex", alignItems: "center", gap: "4px",
                                transition: "var(--transition-fast)",
                                padding: "5px 10px",
                                background: "rgba(99,102,241,0.07)",
                                borderRadius: "8px",
                                border: "1px solid rgba(99,102,241,0.15)",
                            }}
                                onMouseEnter={e => {
                                    e.currentTarget.style.background = "var(--primary)";
                                    e.currentTarget.style.color = "white";
                                }}
                                onMouseLeave={e => {
                                    e.currentTarget.style.background = "rgba(99,102,241,0.07)";
                                    e.currentTarget.style.color = "var(--primary)";
                                }}
                            >
                                View all <span style={{ fontSize: "14px", lineHeight: 1 }}>&rarr;</span>
                            </Link>
                        </div>
                        <div style={{ padding: "12px 16px 16px" }}>
                            <PredictionHistory limit={5} />
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}

export default Dashboard;