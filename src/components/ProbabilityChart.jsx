import React, { useState, useEffect, useRef } from "react";

/* ── Animated Probability Bar ──────────────────────────────────────────── */
function ProbabilityBar({ label, value, color, isActive, delay = 0 }) {
    const pct     = Math.min(100, Math.max(0, value * 100));
    const fillRef = useRef(null);

    useEffect(() => {
        const t = setTimeout(() => {
            if (fillRef.current) fillRef.current.style.width = `${pct}%`;
        }, delay);
        return () => clearTimeout(t);
    }, [pct, delay]);

    return (
        <div style={{ marginBottom: "18px" }}>
            {/* Label + value row */}
            <div style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "center", marginBottom: "8px",
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{
                        fontSize: "13px", fontWeight: 700,
                        color: isActive ? color : "var(--text-muted)",
                        transition: "color 0.3s ease",
                    }}>
                        {label}
                    </span>
                    {isActive && (
                        <span style={{
                            fontSize: "9px", padding: "2px 8px",
                            borderRadius: "99px",
                            background: `${color}18`,
                            color, border: `1px solid ${color}40`,
                            fontWeight: 800, letterSpacing: "0.5px",
                            textTransform: "uppercase",
                            boxShadow: `0 0 6px ${color}30`,
                        }}>
                            ACTIVE
                        </span>
                    )}
                </div>
                <span style={{
                    fontSize: "16px", fontWeight: 800,
                    color: isActive ? color : "var(--text-faint)",
                    fontFamily: "'JetBrains Mono', monospace",
                    letterSpacing: "-0.5px",
                    transition: "color 0.3s ease",
                }}>
                    {pct.toFixed(1)}%
                </span>
            </div>

            {/* Track */}
            <div style={{
                background: "rgba(148,163,184,0.08)",
                borderRadius: "99px", height: "10px",
                overflow: "hidden",
                border: isActive ? `1px solid ${color}25` : "1px solid transparent",
                position: "relative",
            }}>
                <div
                    ref={fillRef}
                    style={{
                        width: "0%", height: "100%",
                        borderRadius: "99px",
                        background: isActive
                            ? `linear-gradient(90deg, ${color}bb, ${color})`
                            : `${color}35`,
                        boxShadow: isActive ? `0 0 12px ${color}55` : "none",
                        transition: "width 1s cubic-bezier(0.4, 0, 0.2, 1)",
                    }}
                />
                {/* Shimmer effect on active bar */}
                {isActive && (
                    <div style={{
                        position: "absolute", inset: 0,
                        background: "linear-gradient(90deg, transparent 30%, rgba(255,255,255,0.12) 50%, transparent 70%)",
                        backgroundSize: "200% 100%",
                        animation: "shimmer 2s infinite",
                        borderRadius: "99px",
                    }} />
                )}
            </div>
        </div>
    );
}

/* ── Gauge Arc (SVG) ─────────────────────────────────────────────────── */
function GaugeArc({ value, color, glow, label }) {
    const r = 48;
    const circ = 2 * Math.PI * r;
    // Half-circle gauge: use 75% of circumference
    const arcFraction = 0.72;
    const arcLength = circ * arcFraction;
    const fill = arcLength * Math.min(value / 100, 1);

    return (
        <div style={{ textAlign: "center", flex: 1 }}>
            <svg width="100" height="60" viewBox="0 0 100 60" style={{ overflow: "visible" }}>
                {/* Background arc */}
                <path
                    d="M 10,55 A 40,40 0 0 1 90,55"
                    fill="none" stroke="rgba(148,163,184,0.08)" strokeWidth="7"
                    strokeLinecap="round"
                />
                {/* Fill arc */}
                <path
                    d="M 10,55 A 40,40 0 0 1 90,55"
                    fill="none" stroke={color} strokeWidth="7"
                    strokeLinecap="round"
                    strokeDasharray={`${(fill / arcLength) * 100} 100`}
                    pathLength="100"
                    style={{
                        filter: `drop-shadow(0 0 4px ${glow})`,
                        transition: "stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)",
                    }}
                />
                {/* Value text */}
                <text x="50" y="50" textAnchor="middle" fontSize="13" fontWeight="800"
                    fill={color} fontFamily="'JetBrains Mono', monospace">
                    {value.toFixed(1)}%
                </text>
            </svg>
            <div style={{ fontSize: "10px", color: "var(--text-faint)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.6px", marginTop: "2px" }}>
                {label}
            </div>
        </div>
    );
}

/* ── Main ProbabilityChart ────────────────────────────────────────────── */
function ProbabilityChart({ prediction }) {
    const [view, setView] = useState("bars"); // "bars" | "gauges"

    if (!prediction) return null;

    const pCardio   = prediction.cardiomegaly ?? 0;
    const pComorbid = prediction.comorbidity  ?? 0;
    const pNormal   = prediction.normal       ?? Math.max(0, 1 - pCardio - pComorbid);

    const activeLabel =
        pCardio  >= 0.50 ? "cardio"   :
        pComorbid > 0.05 ? "comorbid" :
        "normal";

    const dominantColor =
        activeLabel === "cardio"   ? "#f87171" :
        activeLabel === "comorbid" ? "#fbbf24" : "#4ade80";

    const dominantGlow =
        activeLabel === "cardio"   ? "rgba(239,68,68,0.4)"  :
        activeLabel === "comorbid" ? "rgba(245,158,11,0.4)" : "rgba(34,197,94,0.4)";

    return (
        <div className="card p-4 border-0 animate-scale-in">

            {/* ── Top neon accent bar ── */}
            <div style={{
                position: "absolute", top: 0, left: 0, right: 0, height: "2px",
                background: `linear-gradient(90deg, ${dominantColor}, transparent 80%)`,
                borderRadius: "16px 16px 0 0",
            }} />

            {/* ── Header ── */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "12px", marginBottom: "18px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{
                        width: "42px", height: "42px", borderRadius: "11px",
                        background: `${dominantColor}12`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        border: `1px solid ${dominantColor}28`,
                        boxShadow: `0 0 12px ${dominantColor}18`,
                    }}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                            fill="none" stroke={dominantColor} strokeWidth="2"
                            strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="20" x2="18" y2="10"/>
                            <line x1="12" y1="20" x2="12" y2="4"/>
                            <line x1="6"  y1="20" x2="6"  y2="14"/>
                        </svg>
                    </div>
                    <div>
                        <h6 style={{ margin: 0, fontWeight: 800, color: "var(--text-main)", fontSize: "14.5px" }}>
                            Classification Probabilities
                        </h6>
                        <p style={{ margin: 0, fontSize: "11px", color: "var(--text-faint)" }}>
                            3-Way: Cardio · Co-morbidity · Normal
                        </p>
                    </div>
                </div>

                {/* View toggle */}
                <div style={{
                    display: "flex", borderRadius: "9px",
                    border: "1px solid var(--border-color)",
                    overflow: "hidden", flexShrink: 0,
                }}>
                    {[
                        { id: "bars",   icon: "≡" },
                        { id: "gauges", icon: "◑" },
                    ].map(btn => (
                        <button key={btn.id} onClick={() => setView(btn.id)} style={{
                            padding: "5px 11px",
                            border: "none",
                            background: view === btn.id ? "rgba(99,102,241,0.15)" : "transparent",
                            color: view === btn.id ? "var(--primary)" : "var(--text-faint)",
                            cursor: "pointer", fontWeight: 700, fontSize: "14px",
                            transition: "all 0.2s",
                        }}>
                            {btn.icon}
                        </button>
                    ))}
                </div>
            </div>

            {/* ── Bar View ── */}
            {view === "bars" && (
                <>
                    <ProbabilityBar
                        label="🔴 Cardiomegaly"
                        value={pCardio}
                        color="#f87171"
                        isActive={activeLabel === "cardio"}
                        delay={80}
                    />
                    <ProbabilityBar
                        label="🟡 Co-morbidity"
                        value={pComorbid}
                        color="#fbbf24"
                        isActive={activeLabel === "comorbid"}
                        delay={220}
                    />
                    <ProbabilityBar
                        label="🟢 No Finding"
                        value={pNormal}
                        color="#4ade80"
                        isActive={activeLabel === "normal"}
                        delay={360}
                    />
                </>
            )}

            {/* ── Gauge Arc View ── */}
            {view === "gauges" && (
                <div style={{ display: "flex", justifyContent: "space-around", padding: "10px 0 6px" }}>
                    <GaugeArc
                        value={pCardio * 100}
                        color="#f87171"
                        glow="rgba(239,68,68,0.5)"
                        label="Cardio"
                    />
                    <GaugeArc
                        value={pComorbid * 100}
                        color="#fbbf24"
                        glow="rgba(245,158,11,0.5)"
                        label="Co-morbid"
                    />
                    <GaugeArc
                        value={pNormal * 100}
                        color="#4ade80"
                        glow="rgba(34,197,94,0.5)"
                        label="Normal"
                    />
                </div>
            )}

            {/* ── Threshold footer ── */}
            <div style={{
                marginTop: "16px",
                padding: "10px 14px",
                borderRadius: "10px",
                background: "rgba(99,102,241,0.04)",
                border: "1px solid var(--border-subtle)",
            }}>
                <p style={{
                    margin: 0, fontSize: "10px",
                    color: "var(--text-faint)",
                    lineHeight: "1.7",
                    fontFamily: "'JetBrains Mono', monospace",
                }}>
                    Threshold: Cardiomegaly ≥ 50% · Co-morbidity ≥ 45% · Normal &lt; 45%
                </p>
            </div>

        </div>
    );
}

export default ProbabilityChart;