import React, { useState } from "react";

/* ── Disease metadata for clinical display ────────────────────────────────── */
const DISEASE_META = {
    "Cardiomegaly":       { color: "#f87171", icon: "🫀", region: "Cardiac Silhouette",  zone: "Centro-inferior mediastinum",      map: "🔴 HOT" },
    "Atelectasis":        { color: "#fb923c", icon: "🫁", region: "Lung Parenchyma",      zone: "Lower lobe / linear bands",        map: "🟠 AUTUMN" },
    "Effusion":           { color: "#38bdf8", icon: "💧", region: "Pleural Space",         zone: "Costophrenic angle / lateral base", map: "🔵 OCEAN" },
    "Infiltration":       { color: "#fb923c", icon: "🌫️", region: "Lung Parenchyma",      zone: "Bilateral mid/lower fields",        map: "🟠 AUTUMN" },
    "Mass":               { color: "#c084fc", icon: "⚫", region: "Focal Opacity",         zone: "Perihilar / lobar",                 map: "🟣 MAGMA" },
    "Nodule":             { color: "#c084fc", icon: "🔵", region: "Focal Nodularity",      zone: "Upper/mid lung field",              map: "🟣 MAGMA" },
    "Pneumonia":          { color: "#fb923c", icon: "🦠", region: "Consolidation",         zone: "Right lower lobe (most common)",    map: "🟠 AUTUMN" },
    "Pneumothorax":       { color: "#7dd3fc", icon: "💨", region: "Pleural Line",          zone: "Lateral thoracic wall",             map: "🌈 RAINBOW" },
    "Consolidation":      { color: "#fdba74", icon: "☁️", region: "Air-space Opacity",    zone: "Lobar / segmental",                 map: "🟠 AUTUMN" },
    "Edema":              { color: "#60a5fa", icon: "💦", region: "Pulmonary Vessels",     zone: "Perihilar / bilateral fields",      map: "🔵 OCEAN" },
    "Emphysema":          { color: "#e879f9", icon: "💭", region: "Hyperinflation",        zone: "Upper lobes / flattened diaphragm", map: "🌸 PINK" },
    "Fibrosis":           { color: "#a8a29e", icon: "🕸️", region: "Interstitium",         zone: "Lower lobe reticulation",           map: "⬜ BONE" },
    "Pleural_Thickening": { color: "#f97316", icon: "📐", region: "Pleural Surface",       zone: "Lateral / apical pleura",           map: "🟠 AUTUMN" },
    "Hernia":             { color: "#f0abfc", icon: "⤵️", region: "Hiatal/Diaphragmatic", zone: "Retrocardiac / mediastinum",        map: "🌺 SPRING" },
    "No Finding":         { color: "#4ade80", icon: "✅", region: "All Lung Fields",       zone: "No significant activation",         map: "🟢 SUMMER" },
};

const DEFAULT_META = { color: "#818cf8", icon: "🔬", region: "Unspecified Region", zone: "—", map: "🌈 JET" };

/* ── Empty State ─────────────────────────────────────────────────────────── */
function EmptyState() {
    return (
        <div className="card border-0" style={{
            minHeight: "280px",
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center",
            textAlign: "center", padding: "40px 24px",
            position: "relative", overflow: "hidden",
        }}>
            <div style={{
                position: "absolute", inset: 0,
                backgroundImage: "radial-gradient(rgba(99,102,241,0.04) 1px, transparent 1px)",
                backgroundSize: "24px 24px", pointerEvents: "none",
            }} />
            <div style={{
                width: "68px", height: "68px", borderRadius: "18px",
                background: "rgba(99,102,241,0.06)",
                border: "1px solid rgba(99,102,241,0.15)",
                display: "flex", alignItems: "center", justifyContent: "center",
                marginBottom: "16px",
            }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24"
                    fill="none" stroke="rgba(99,102,241,0.3)" strokeWidth="1.5"
                    strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    <line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
                </svg>
            </div>
            <h6 style={{ fontWeight: 800, color: "var(--text-secondary)", marginBottom: "7px", fontSize: "15px" }}>
                Grad-CAM++ Heatmap
            </h6>
            <p style={{ fontSize: "12px", color: "var(--text-faint)", margin: 0, lineHeight: 1.7, maxWidth: "220px" }}>
                Disease-specific activation map will appear here after analysis
            </p>
        </div>
    );
}

/* ── Main HeatmapViewer ──────────────────────────────────────────────────── */
function HeatmapViewer({ heatmap, label }) {
    const [imgError, setImgError] = useState(false);
    const [showOriginal, setShowOriginal] = useState(false);

    if (!heatmap) return <EmptyState />;

    const meta = DISEASE_META[label] || DEFAULT_META;
    const isNormal = label === "No Finding";

    return (
        <div className="card p-4 border-0 animate-scale-in">

            {/* ── Header ── */}
            <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", marginBottom: "14px" }}>
                <div style={{
                    width: "42px", height: "42px", borderRadius: "11px",
                    background: `${meta.color}12`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    border: `1px solid ${meta.color}30`,
                    boxShadow: `0 0 12px ${meta.color}20`,
                    fontSize: "18px", flexShrink: 0,
                }}>
                    {meta.icon}
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "3px" }}>
                        <h6 style={{ margin: 0, fontWeight: 800, color: "var(--text-main)", fontSize: "14.5px" }}>
                            Grad-CAM++ Analysis
                        </h6>
                        <span style={{
                            padding: "2px 8px", borderRadius: "99px",
                            background: `${meta.color}12`, color: meta.color,
                            fontSize: "9.5px", fontWeight: 700,
                            border: `1px solid ${meta.color}30`,
                            letterSpacing: "0.5px", textTransform: "uppercase",
                        }}>
                            {label || "AI XAI"}
                        </span>
                    </div>
                    <p style={{ margin: 0, fontSize: "11px", color: "var(--text-faint)" }}>
                        Gradient-weighted Class Activation Map (Grad-CAM++)
                    </p>
                </div>
            </div>

            {/* ── Anatomical Findings Strip ── */}
            <div style={{
                padding: "12px 14px",
                borderRadius: "12px",
                background: isNormal ? "rgba(34,197,94,0.05)" : `${meta.color}07`,
                border: `1px solid ${meta.color}20`,
                marginBottom: "14px",
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "8px",
            }}>
                {[
                    { label: "Region of Interest",   value: meta.region },
                    { label: "Anatomical Zone",       value: meta.zone   },
                    { label: "Activation Colormap",   value: meta.map    },
                    { label: "XAI Algorithm",         value: "Grad-CAM++ (2018)" },
                ].map(row => (
                    <div key={row.label}>
                        <div style={{ fontSize: "9px", fontWeight: 700, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.8px" }}>
                            {row.label}
                        </div>
                        <div style={{ fontSize: "11.5px", fontWeight: 600, color: meta.color, marginTop: "2px" }}>
                            {row.value}
                        </div>
                    </div>
                ))}
            </div>

            {/* ── Heatmap Viewport ── */}
            <div style={{
                position: "relative",
                borderRadius: "14px",
                overflow: "hidden",
                background: "linear-gradient(160deg, #08111f 0%, #0f1e35 100%)",
                border: `1px solid ${meta.color}30`,
                minHeight: "240px",
                display: "flex", alignItems: "center", justifyContent: "center",
                boxShadow: `inset 0 0 40px rgba(0,0,0,0.4), 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px ${meta.color}15`,
            }}>
                {/* Corner scanner brackets with disease color */}
                {["tl","tr","bl","br"].map(pos => {
                    const isTop  = pos.startsWith("t");
                    const isLeft = pos.endsWith("l");
                    return (
                        <div key={pos} style={{
                            position: "absolute",
                            [isTop  ? "top"    : "bottom"]: "10px",
                            [isLeft ? "left"   : "right" ]: "10px",
                            width: "20px", height: "20px",
                            borderTop:    isTop  ? `2px solid ${meta.color}cc` : "none",
                            borderBottom: !isTop  ? `2px solid ${meta.color}cc` : "none",
                            borderLeft:   isLeft  ? `2px solid ${meta.color}cc` : "none",
                            borderRight:  !isLeft ? `2px solid ${meta.color}cc` : "none",
                            zIndex: 5, borderRadius: "2px",
                            boxShadow: `0 0 6px ${meta.color}55`,
                        }} />
                    );
                })}

                {/* Disease label badge */}
                <div style={{
                    position: "absolute", top: "12px", left: "12px", zIndex: 10,
                    display: "flex", flexDirection: "column", gap: "5px",
                }}>
                    {/* XAI badge */}
                    <span style={{
                        background: "rgba(8,17,31,0.88)",
                        border: `1px solid ${meta.color}50`,
                        backdropFilter: "blur(8px)",
                        borderRadius: "8px", padding: "4px 10px",
                        fontSize: "10px", fontWeight: 700,
                        letterSpacing: "0.8px",
                        color: meta.color,
                        display: "flex", alignItems: "center", gap: "6px",
                    }}>
                        <span style={{
                            width: "6px", height: "6px", borderRadius: "50%",
                            background: "#22c55e",
                            boxShadow: "0 0 6px rgba(34,197,94,0.8)",
                            animation: "livePulse 2s infinite",
                        }} />
                        GRAD-CAM++
                    </span>

                    {/* Region tag */}
                    {label && (
                        <span style={{
                            background: "rgba(8,17,31,0.82)",
                            border: `1px solid ${meta.color}35`,
                            backdropFilter: "blur(6px)",
                            borderRadius: "8px", padding: "3px 9px",
                            fontSize: "9.5px", fontWeight: 700,
                            color: meta.color,
                            letterSpacing: "0.3px",
                        }}>
                            {meta.icon} {meta.region}
                        </span>
                    )}
                </div>

                {/* Original/Heatmap toggle button */}
                <div style={{ position: "absolute", top: "12px", right: "12px", zIndex: 10 }}>
                    <button
                        onClick={() => setShowOriginal(v => !v)}
                        style={{
                            background: "rgba(8,17,31,0.85)",
                            border: `1px solid ${meta.color}40`,
                            backdropFilter: "blur(8px)",
                            borderRadius: "8px", padding: "5px 10px",
                            fontSize: "10px", fontWeight: 700,
                            color: meta.color,
                            cursor: "pointer",
                            display: "flex", alignItems: "center", gap: "5px",
                            transition: "all 0.2s",
                        }}
                        onMouseEnter={e => { e.currentTarget.style.background = `${meta.color}20`; }}
                        onMouseLeave={e => { e.currentTarget.style.background = "rgba(8,17,31,0.85)"; }}
                    >
                        {showOriginal ? (
                            <>
                                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" fill="none"
                                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <circle cx="6" cy="6" r="5"/><circle cx="6" cy="6" r="2"/>
                                </svg>
                                CAM View
                            </>
                        ) : (
                            <>
                                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" fill="none"
                                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <rect x="2" y="2" width="9" height="9" rx="1"/>
                                    <circle cx="4.5" cy="4.5" r="0.8" fill="currentColor"/>
                                    <polyline points="9 7 7 5 5 7 2 9"/>
                                </svg>
                                Original
                            </>
                        )}
                    </button>
                </div>

                {imgError ? (
                    <div style={{ textAlign: "center", padding: "32px" }}>
                        <div style={{ fontSize: "32px", marginBottom: "8px" }}>⚠️</div>
                        <p style={{ color: "#94a3b8", fontSize: "12px", margin: 0 }}>
                            Heatmap generation failed
                        </p>
                    </div>
                ) : (
                    <>
                        <img
                            src={heatmap}
                            alt={`Grad-CAM++ Activation Heatmap — ${label || "AI Analysis"}`}
                            style={{
                                position: "absolute",
                                top: 0, left: 0, right: 0, bottom: 0,
                                width: "100%", height: "100%",
                                maxHeight: "290px",
                                objectFit: "contain",
                                display: "block",
                                opacity: showOriginal ? 0 : 1,
                                transition: "opacity 0.4s ease",
                            }}
                            onError={() => setImgError(true)}
                        />
                        {/* Scan line */}
                        <div style={{
                            position: "absolute", top: 0, left: 0, right: 0,
                            height: "2px",
                            background: `linear-gradient(90deg, transparent, ${meta.color}cc, transparent)`,
                            boxShadow: `0 0 10px ${meta.color}88`,
                            animation: "scanMove 3s linear infinite",
                            zIndex: 4,
                        }} />
                        {/* Placeholder for image height */}
                        <div style={{ height: "240px", width: "100%" }} />
                    </>
                )}
            </div>

            {/* ── Colormap Legend ── */}
            <div style={{ marginTop: "16px" }}>
                <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    fontSize: "10px", fontWeight: 700, textTransform: "uppercase",
                    letterSpacing: "0.5px", color: "var(--text-faint)", marginBottom: "7px",
                }}>
                    <span>Background (Suppressed)</span>
                    <span style={{ color: meta.color }}>Peak Activation — {meta.region}</span>
                </div>

                {/* JET-style gradient bar */}
                <div style={{
                    height: "8px", borderRadius: "99px",
                    background: "linear-gradient(90deg, #00007f, #0000ff, #00ffff, #00ff00, #ffff00, #ff7f00, #ff0000)",
                    boxShadow: `0 2px 10px rgba(0,0,0,0.2), 0 0 8px ${meta.color}20`,
                }} />

                {/* Legend chips */}
                <div style={{
                    display: "flex", justifyContent: "center",
                    gap: "16px", marginTop: "10px",
                    flexWrap: "wrap",
                }}>
                    {[
                        { color: meta.color,   label: `${label || "Finding"} Focus` },
                        { color: "#facc15",    label: "Moderate Activation" },
                        { color: "#60a5fa",    label: "Low Activation" },
                        { color: "#1d4ed8",    label: "Suppressed" },
                    ].map(chip => (
                        <span key={chip.label} style={{
                            display: "flex", alignItems: "center", gap: "5px",
                            fontSize: "10.5px", color: "var(--text-faint)",
                        }}>
                            <span style={{
                                width: "9px", height: "9px", borderRadius: "50%",
                                background: chip.color, display: "inline-block", flexShrink: 0,
                                boxShadow: `0 0 4px ${chip.color}66`,
                            }} />
                            {chip.label}
                        </span>
                    ))}
                </div>
            </div>

            {/* ── Clinical Interpretation Note ── */}
            <div style={{
                marginTop: "14px", padding: "11px 14px",
                borderRadius: "10px",
                background: "rgba(99,102,241,0.04)",
                border: "1px solid rgba(99,102,241,0.1)",
            }}>
                <div style={{ fontSize: "9px", fontWeight: 700, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.8px", marginBottom: "4px" }}>
                    Clinical XAI Note
                </div>
                <p style={{ margin: 0, fontSize: "11px", color: "var(--text-muted)", lineHeight: 1.6 }}>
                    {isNormal
                        ? "No significant pathological activation detected. The model shows diffuse low-energy activations distributed across all fields — consistent with a clear X-ray."
                        : `The Grad-CAM++ map highlights the neural attention on <strong>${meta.region}</strong> (${meta.zone}). Red/hot regions have the highest gradient-weighted activation driving the <em>${label}</em> prediction.`
                    }
                </p>
            </div>

        </div>
    );
}

export default HeatmapViewer;