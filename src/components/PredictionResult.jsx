import React from "react";

// ── 3-Way Classification Labels ──────────────────────────────────────────
const NIH_DISEASES = [
    "Atelectasis","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
];

function getClassificationBranch(label) {
    if (label === "Cardiomegaly") return "cardio";
    if (label === "No Finding" || label.toLowerCase() === "normal") return "normal";
    if (NIH_DISEASES.includes(label)) return "comorbid";
    return "normal";
}

/* ── Theme Map ──────────────────────────────────────────────────────────── */
const THEMES = {
    cardio:   {
        bar:    "#f87171",
        glow:   "rgba(239,68,68,0.25)",
        light:  "rgba(239,68,68,0.06)",
        border: "rgba(239,68,68,0.2)",
        label:  "Cardiomegaly Detected",
        emoji:  "🔴",
        action: "Immediate Action Required",
    },
    comorbid: {
        bar:    "#fbbf24",
        glow:   "rgba(245,158,11,0.2)",
        light:  "rgba(245,158,11,0.06)",
        border: "rgba(245,158,11,0.2)",
        label:  "Co-morbidity Found",
        emoji:  "🟡",
        action: "Follow-up Recommended",
    },
    normal:   {
        bar:    "#4ade80",
        glow:   "rgba(34,197,94,0.2)",
        light:  "rgba(34,197,94,0.06)",
        border: "rgba(34,197,94,0.2)",
        label:  "No Finding",
        emoji:  "🟢",
        action: "All Clear",
    },
};

/* ── Confidence Ring (SVG arc) ──────────────────────────────────────────── */
function ConfidenceRing({ conf, color, glow }) {
    const r = 44;
    const circ = 2 * Math.PI * r;
    const offset = circ - (conf / 100) * circ;

    return (
        <div className="result-confidence-ring">
            <svg width="100" height="100" viewBox="0 0 100 100" style={{ transform: "rotate(-90deg)" }}>
                <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(148,163,184,0.08)" strokeWidth="6" />
                <circle
                    cx="50" cy="50" r={r} fill="none"
                    stroke={color} strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={circ}
                    strokeDashoffset={offset}
                    style={{
                        filter: `drop-shadow(0 0 6px ${glow})`,
                        transition: "stroke-dashoffset 1.4s cubic-bezier(0.4,0,0.2,1)",
                    }}
                />
            </svg>
            <div className="ring-label">
                <span style={{
                    fontSize: "18px", fontWeight: 900,
                    color: color,
                    fontFamily: "'JetBrains Mono', monospace",
                    lineHeight: 1,
                }}>
                    {conf.toFixed(0)}%
                </span>
                <span style={{ fontSize: "9px", color: "var(--text-faint)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginTop: "2px" }}>
                    Conf.
                </span>
            </div>
        </div>
    );
}

/* ── Empty State ────────────────────────────────────────────────────────── */
function EmptyState() {
    return (
        <div className="card border-0" style={{
            minHeight: "280px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            textAlign: "center",
            padding: "40px 24px",
            position: "relative",
            overflow: "hidden",
        }}>
            {/* Background pattern */}
            <div style={{
                position: "absolute", inset: 0,
                backgroundImage: "radial-gradient(rgba(99,102,241,0.04) 1px, transparent 1px)",
                backgroundSize: "24px 24px",
                pointerEvents: "none",
            }} />

            <div style={{
                width: "68px", height: "68px", borderRadius: "18px",
                background: "rgba(99,102,241,0.06)",
                border: "1px solid rgba(99,102,241,0.12)",
                display: "flex", alignItems: "center", justifyContent: "center",
                marginBottom: "18px",
                position: "relative",
            }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24"
                    fill="none" stroke="rgba(99,102,241,0.3)" strokeWidth="1.5"
                    strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </div>
            <h6 style={{ fontWeight: 800, color: "var(--text-secondary)", marginBottom: "7px", fontSize: "15px" }}>
                AI Diagnosis Output
            </h6>
            <p style={{ fontSize: "12px", color: "var(--text-faint)", margin: 0, lineHeight: 1.7, maxWidth: "220px" }}>
                Upload a chest X-ray and click <strong style={{ color: "var(--primary)" }}>Analyze</strong> to receive the AI diagnosis
            </p>
        </div>
    );
}

/* ── Main PredictionResult ──────────────────────────────────────────────── */
function PredictionResult({ result }) {
    if (!result) return <EmptyState />;

    const branch = getClassificationBranch(result.label);
    const t      = THEMES[branch];
    const conf   = result.confidence * 100;

    return (
        <div className="card border-0 animate-scale-in" style={{ overflow: "hidden", position: "relative" }}>

            {/* Neon top accent bar */}
            <div style={{
                position: "absolute", top: 0, left: 0, right: 0,
                height: "3px",
                background: `linear-gradient(90deg, ${t.bar}, ${t.bar}88, transparent)`,
                boxShadow: `0 0 12px ${t.glow}`,
            }} />

            {/* Background glow */}
            <div style={{
                position: "absolute", top: -60, right: -60,
                width: 200, height: 200, borderRadius: "50%",
                background: `radial-gradient(circle, ${t.glow} 0%, transparent 70%)`,
                pointerEvents: "none",
            }} />

            <div style={{ padding: "24px", position: "relative", zIndex: 1 }}>

                {/* ── Header ── */}
                <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "22px" }}>
                    <div style={{
                        width: "42px", height: "42px", borderRadius: "11px",
                        background: t.light, color: t.bar,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        border: `1px solid ${t.border}`,
                        boxShadow: `0 0 14px ${t.glow}`,
                    }}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24"
                            fill="none" stroke="currentColor" strokeWidth="2"
                            strokeLinecap="round" strokeLinejoin="round">
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                        </svg>
                    </div>
                    <div>
                        <h6 style={{ margin: 0, fontWeight: 800, color: "var(--text-main)", fontSize: "15px" }}>
                            AI Diagnosis Output
                        </h6>
                        <p style={{ margin: 0, fontSize: "11px", color: "var(--text-faint)" }}>
                            ConvNeXt-V2-Tiny · 3-Way Classification
                        </p>
                    </div>
                    <div style={{ marginLeft: "auto" }}>
                        <span style={{
                            padding: "4px 12px",
                            borderRadius: "99px",
                            background: t.light,
                            color: t.bar,
                            border: `1px solid ${t.border}`,
                            fontSize: "10.5px",
                            fontWeight: 700,
                            letterSpacing: "0.5px",
                            textTransform: "uppercase",
                        }}>
                            {t.action}
                        </span>
                    </div>
                </div>

                {/* ── Classification Banner ── */}
                <div style={{
                    padding: "20px",
                    borderRadius: "14px",
                    background: t.light,
                    border: `1px solid ${t.border}`,
                    marginBottom: "20px",
                    display: "flex",
                    alignItems: "center",
                    gap: "20px",
                }}>
                    {/* Confidence Ring */}
                    <ConfidenceRing conf={conf} color={t.bar} glow={t.glow} />

                    {/* Label + bar */}
                    <div style={{ flex: 1 }}>
                        <div style={{ fontSize: "28px", marginBottom: "6px" }}>{t.emoji}</div>
                        <div style={{
                            fontSize: "10px", fontWeight: 700, textTransform: "uppercase",
                            letterSpacing: "0.8px", color: "var(--text-faint)", marginBottom: "4px",
                        }}>
                            Classification Result
                        </div>
                        <h2 style={{
                            margin: "0 0 10px",
                            fontWeight: 900, color: "var(--text-main)",
                            fontSize: "20px", letterSpacing: "-0.5px",
                        }}>
                            {result.label}
                        </h2>
                        {/* Slim progress bar */}
                        <div style={{
                            height: "5px", borderRadius: "99px",
                            background: "rgba(148,163,184,0.08)",
                            overflow: "hidden",
                        }}>
                            <div style={{
                                width: `${conf}%`, height: "100%", borderRadius: "99px",
                                background: `linear-gradient(90deg, ${t.bar}cc, ${t.bar})`,
                                boxShadow: `0 0 8px ${t.glow}`,
                                transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
                            }} />
                        </div>
                    </div>
                </div>

                {/* ── Branch Pills ── */}
                <div style={{ display: "flex", gap: "7px", marginBottom: "20px", justifyContent: "center" }}>
                    {[
                        { id: "cardio",   emoji: "🔴", text: "Cardiomegaly", color: "#f87171" },
                        { id: "comorbid", emoji: "🟡", text: "Co-morbidity",  color: "#fbbf24" },
                        { id: "normal",   emoji: "🟢", text: "No Finding",    color: "#4ade80" },
                    ].map(b => (
                        <span key={b.id} style={{
                            fontSize: "11px", padding: "5px 13px",
                            borderRadius: "99px", fontWeight: 600,
                            background: branch === b.id ? `${b.color}12` : "rgba(148,163,184,0.05)",
                            color:      branch === b.id ? b.color : "var(--text-faint)",
                            border:     `1px solid ${branch === b.id ? `${b.color}35` : "rgba(148,163,184,0.1)"}`,
                            transition: "all 0.25s ease",
                            boxShadow:  branch === b.id ? `0 0 8px ${b.color}20` : "none",
                        }}>
                            {b.emoji} {b.text}
                        </span>
                    ))}
                </div>

                {/* ── Clinical Guidance ── */}
                <div style={{
                    padding: "16px 18px",
                    borderRadius: "12px",
                    background: t.light,
                    border: `1px solid ${t.border}`,
                }}>
                    {branch === "cardio" && (
                        <>
                            <p style={{ fontSize: "13px", fontWeight: 700, color: t.bar, marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
                                🏥 Immediate Cardiology Consultation Required
                            </p>
                            <p style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "12px", lineHeight: "1.6" }}>
                                Cardiomegaly (enlarged heart) detected. Immediate specialist review is strongly advised.
                            </p>
                            <ul style={{ margin: "0 0 12px", paddingLeft: "16px", lineHeight: "1.9", fontSize: "12px", color: "var(--text-main)" }}>
                                <li>
                                    <strong>Apollo Heart Institutes</strong>&nbsp;—&nbsp;
                                    <a href="https://www.google.com/maps/search/Apollo+Heart+Institutes" target="_blank" rel="noreferrer"
                                        style={{ color: "var(--primary)", fontWeight: 700, textDecoration: "none" }}>
                                        Get Directions (2.4 km)
                                    </a>
                                </li>
                                <li>
                                    <strong>Fortis Escorts Heart Institute</strong>&nbsp;—&nbsp;
                                    <a href="https://www.google.com/maps/search/Fortis+Escorts+Heart+Institute" target="_blank" rel="noreferrer"
                                        style={{ color: "var(--primary)", fontWeight: 700, textDecoration: "none" }}>
                                        Get Directions (5.1 km)
                                    </a>
                                </li>
                            </ul>
                            <a href="tel:911" style={{
                                display: "flex", alignItems: "center", justifyContent: "center", gap: "7px",
                                width: "100%", padding: "10px",
                                borderRadius: "10px",
                                background: "rgba(239,68,68,0.1)",
                                border: "1px solid rgba(239,68,68,0.25)",
                                color: "#f87171", fontWeight: 700, fontSize: "13px",
                                textDecoration: "none",
                                transition: "all 0.2s",
                            }}
                                onMouseEnter={e => { e.currentTarget.style.background = "rgba(239,68,68,0.16)"; }}
                                onMouseLeave={e => { e.currentTarget.style.background = "rgba(239,68,68,0.1)"; }}
                            >
                                🚨 Contact Emergency Intake (911)
                            </a>
                        </>
                    )}

                    {branch === "comorbid" && (
                        <>
                            <p style={{ fontSize: "13px", fontWeight: 700, color: t.bar, marginBottom: "8px" }}>
                                ⚠️ Pulmonary Co-morbidity Detected
                            </p>
                            <p style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "10px", lineHeight: "1.6" }}>
                                No cardiomegaly found. Pulmonary abnormalities identified above the detection threshold (≥ 45%):
                            </p>
                            {result.top_findings?.length > 0 ? (
                                result.top_findings.map(([name, prob], i) => (
                                    <div key={name} style={{
                                        display: "flex", justifyContent: "space-between",
                                        alignItems: "center", marginBottom: "8px",
                                    }}>
                                        <span style={{ fontWeight: 600, fontSize: "12px", color: "var(--text-secondary)" }}>
                                            {i === 0 ? "🥇" : i === 1 ? "🥈" : "🥉"} {name}
                                        </span>
                                        <span style={{
                                            fontWeight: 700, fontSize: "12px", color: t.bar,
                                            fontFamily: "'JetBrains Mono', monospace",
                                        }}>
                                            {(prob * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))
                            ) : (
                                <p style={{ fontSize: "12px", color: "var(--text-muted)", margin: 0 }}>
                                    Primary finding: <strong>{result.label}</strong>
                                </p>
                            )}
                            <p style={{ fontSize: "11px", color: "var(--text-faint)", marginTop: "10px", marginBottom: 0, lineHeight: "1.5" }}>
                                Follow-up with a pulmonologist or general physician is recommended.
                            </p>
                        </>
                    )}

                    {branch === "normal" && (
                        <>
                            <p style={{ fontSize: "13px", fontWeight: 700, color: t.bar, marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
                                ✅ No Significant Pathology Found
                            </p>
                            <p style={{ fontSize: "12px", color: "var(--text-muted)", margin: 0, lineHeight: "1.7" }}>
                                The X-ray appears clear with no significant cardiomegaly or pulmonary abnormalities
                                detected in the frontal plane. Maintain a healthy lifestyle and proceed with
                                regular annual checkups.
                            </p>
                        </>
                    )}
                </div>

            </div>
        </div>
    );
}

export default PredictionResult;