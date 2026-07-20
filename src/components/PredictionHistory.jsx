import React, { useEffect, useState } from "react";
import API from "../services/api";

const FILTERS = ["All", "Cardiomegaly", "No Finding", "Other Disease"];

const FILTER_COLORS = {
    All:             { color: "var(--primary)",   bg: "rgba(99,102,241,0.1)" },
    Cardiomegaly:    { color: "#f87171",          bg: "rgba(239,68,68,0.1)"  },
    "No Finding":    { color: "#4ade80",          bg: "rgba(34,197,94,0.1)"  },
    "Other Disease": { color: "#fbbf24",          bg: "rgba(245,158,11,0.1)" },
};

function getRowStyle(prediction) {
    const isCardio    = prediction === "Cardiomegaly";
    const isNoFinding = prediction === "No Finding" || prediction === "Normal";
    const color       = isCardio ? "#f87171" : isNoFinding ? "#4ade80" : "#fbbf24";
    const bg          = isCardio ? "rgba(239,68,68,0.06)" : isNoFinding ? "rgba(34,197,94,0.06)" : "rgba(245,158,11,0.06)";
    const border      = isCardio ? "rgba(239,68,68,0.25)"  : isNoFinding ? "rgba(34,197,94,0.25)"  : "rgba(245,158,11,0.25)";
    const icon        = isCardio ? "🔴" : isNoFinding ? "🟢" : "🟡";
    return { color, bg, border, icon };
}

/* ── Skeleton Loader ── */
function HistorySkeleton() {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {[1, 2, 3].map(i => (
                <div key={i} style={{
                    display: "flex", alignItems: "center", gap: "12px",
                    padding: "12px 14px", borderRadius: "14px",
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid var(--border-color)",
                }}>
                    <div className="skeleton-box" style={{ width: "48px", height: "48px", borderRadius: "9px" }} />
                    <div style={{ flex: 1 }}>
                        <div className="skeleton-box" style={{ width: "120px", height: "16px", borderRadius: "4px", marginBottom: "8px" }} />
                        <div className="skeleton-box" style={{ width: "80px", height: "12px", borderRadius: "4px" }} />
                    </div>
                    <div style={{ width: "64px" }}>
                        <div className="skeleton-box" style={{ width: "40px", height: "16px", borderRadius: "4px", marginLeft: "auto", marginBottom: "8px" }} />
                        <div className="skeleton-box" style={{ width: "64px", height: "4px", borderRadius: "2px" }} />
                    </div>
                </div>
            ))}
        </div>
    );
}

function PredictionHistory({ limit = null }) {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filter,  setFilter]  = useState("All");

    useEffect(() => {
        API.get("/prediction-history")
            .then(res => { setHistory(res.data); setLoading(false); })
            .catch(() => { setLoading(false); });
    }, []);

    const filtered = history
        .filter(item => {
            if (filter === "All") return true;
            if (filter === "Cardiomegaly")  return item.prediction === "Cardiomegaly";
            if (filter === "No Finding")    return item.prediction === "No Finding" || item.prediction === "Normal";
            return item.prediction !== "Cardiomegaly" && item.prediction !== "No Finding" && item.prediction !== "Normal";
        })
        .slice(0, limit ?? undefined);

    if (loading) {
        return <HistorySkeleton />;
    }

    return (
        <div className="animate-fade-up">

            {/* ── Filter chips (only on full page) ── */}
            {!limit && (
                <div style={{ display: "flex", gap: "8px", marginBottom: "20px", flexWrap: "wrap", alignItems: "center" }}>
                    {FILTERS.map(f => {
                        const fc = FILTER_COLORS[f];
                        const active = filter === f;
                        return (
                            <button key={f} onClick={() => setFilter(f)} style={{
                                padding: "6px 16px",
                                borderRadius: "99px",
                                border: `1px solid ${active ? fc.color + '55' : 'var(--border-color)'}`,
                                background: active ? fc.bg : "transparent",
                                color: active ? fc.color : "var(--text-faint)",
                                fontWeight: 700, fontSize: "12px",
                                cursor: "pointer", transition: "var(--transition-fast)",
                                display: "flex", alignItems: "center", gap: "6px",
                            }}>
                                {active && <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: fc.color, boxShadow: `0 0 6px ${fc.color}` }} />}
                                {f}
                            </button>
                        );
                    })}
                    <span style={{ marginLeft: "auto", color: "var(--text-faint)", fontSize: "12px", fontWeight: 700 }}>
                        {filtered.length} Record{filtered.length !== 1 ? "s" : ""}
                    </span>
                </div>
            )}

            {/* ── Empty state ── */}
            {filtered.length === 0 && (
                <div style={{
                    textAlign: "center", padding: "40px 16px",
                    background: "rgba(255,255,255,0.02)",
                    borderRadius: "14px", border: "1px dashed var(--border-color)",
                }}>
                    <div style={{ fontSize: "32px", marginBottom: "12px", filter: "grayscale(1) opacity(0.5)" }}>🩻</div>
                    <div style={{ fontWeight: 800, marginBottom: "4px", color: "var(--text-secondary)", fontSize: "14px" }}>
                        No records found
                    </div>
                    <div style={{ fontSize: "12px", color: "var(--text-faint)" }}>
                        {filter === "All" ? "Upload a chest X-ray to see history here." : `No predictions match the '${filter}' filter.`}
                    </div>
                </div>
            )}

            {/* ── Records ── */}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {filtered.map((item, idx) => {
                    const { color, bg, border, icon } = getRowStyle(item.prediction);
                    const conf = (item.confidence * 100).toFixed(1);

                    return (
                        <div
                            key={item.id || idx}
                            className="history-row"
                            style={{
                                display: "flex", alignItems: "center", gap: "14px",
                                padding: "12px 14px",
                                borderRadius: "14px",
                                border: "1px solid var(--border-color)",
                                background: "rgba(255,255,255,0.02)",
                                transition: "all 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
                                cursor: "pointer",
                                position: "relative",
                                overflow: "hidden",
                            }}
                            onMouseEnter={e => {
                                e.currentTarget.style.background = bg;
                                e.currentTarget.style.borderColor = border;
                                e.currentTarget.style.transform = "translateY(-2px)";
                                e.currentTarget.style.boxShadow = `0 8px 24px rgba(0,0,0,0.3), inset 0 0 0 1px ${border}`;
                            }}
                            onMouseLeave={e => {
                                e.currentTarget.style.background = "rgba(255,255,255,0.02)";
                                e.currentTarget.style.borderColor = "var(--border-color)";
                                e.currentTarget.style.transform = "translateY(0)";
                                e.currentTarget.style.boxShadow = "none";
                            }}
                        >
                            {/* Hover accent bar */}
                            <div className="hover-accent" style={{
                                position: "absolute", left: 0, top: 0, bottom: 0, width: "3px",
                                background: color, opacity: 0, transition: "opacity 0.25s",
                            }} />

                            {/* Thumbnail */}
                            <div style={{
                                width: "48px", height: "48px", borderRadius: "10px",
                                background: "rgba(0,0,0,0.2)",
                                border: "1px solid var(--border-subtle)",
                                overflow: "hidden", flexShrink: 0,
                                position: "relative",
                            }}>
                                <img
                                    src={item.image_url}
                                    alt="X-ray heatmap"
                                    onError={e => { e.target.style.display = "none"; }}
                                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                                />
                            </div>

                            {/* Info */}
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                                    <span style={{
                                        background: bg, color, border: `1px solid ${border}`,
                                        padding: "2px 10px", borderRadius: "99px",
                                        fontSize: "11px", fontWeight: 700,
                                        boxShadow: `0 0 8px ${color}15`,
                                    }}>
                                        {icon} {item.prediction}
                                    </span>
                                    <span style={{
                                        fontSize: "10px", fontWeight: 700,
                                        fontFamily: "'JetBrains Mono', monospace",
                                        color: "var(--text-faint)",
                                        background: "rgba(255,255,255,0.05)",
                                        padding: "2px 6px", borderRadius: "4px",
                                    }}>
                                        #{item.id}
                                    </span>
                                </div>
                                <div style={{ fontSize: "11px", color: "var(--text-faint)", display: "flex", alignItems: "center", gap: "6px" }}>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                                    </svg>
                                    {item.date}
                                </div>
                            </div>

                            {/* Confidence bar */}
                            <div style={{ textAlign: "right", flexShrink: 0, width: "70px" }}>
                                <div style={{
                                    fontSize: "16px", fontWeight: 800, color,
                                    fontFamily: "'JetBrains Mono', monospace",
                                    letterSpacing: "-0.5px", marginBottom: "4px",
                                }}>
                                    {conf}%
                                </div>
                                <div style={{
                                    width: "100%", height: "4px",
                                    background: "var(--border-subtle)",
                                    borderRadius: "99px", overflow: "hidden",
                                }}>
                                    <div style={{
                                        width: `${conf}%`, height: "100%",
                                        borderRadius: "99px", background: color,
                                        boxShadow: `0 0 8px ${color}88`,
                                    }}/>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Hover effect injection for accents */}
            <style>{`
                .history-row:hover .hover-accent { opacity: 1 !important; }
            `}</style>
        </div>
    );
}

export default PredictionHistory;