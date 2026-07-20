import React, { useState } from "react";

/* ─── Login Page — Premium Medical AI Login ────────────────────────────────── */
function Login({ onLogin }) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        await new Promise(r => setTimeout(r, 800));
        if (username === "suman" && password === "viit123") {
            onLogin();
        } else {
            setLoading(false);
            setError("Invalid username or password.");
        }
    };

    return (
        <>
            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
                @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');

                .login-page {
                    min-height: 100vh;
                    display: flex;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                    background: #060d1a; /* Matches dashboard deep space dark */
                    overflow: hidden;
                    position: relative;
                }

                /* ── Animated mesh gradient background ── */
                .login-bg {
                    position: absolute;
                    inset: 0;
                    z-index: 0;
                    overflow: hidden;
                }
                .login-bg::before {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: 
                        radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.25) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 20%, rgba(34, 211, 238, 0.15) 0%, transparent 50%),
                        radial-gradient(ellipse at 40% 80%, rgba(79, 70, 229, 0.15) 0%, transparent 50%),
                        radial-gradient(ellipse at 70% 70%, rgba(14, 165, 233, 0.1) 0%, transparent 40%);
                    animation: meshRotate 25s ease-in-out infinite;
                }
                @keyframes meshRotate {
                    0%, 100% { transform: rotate(0deg) scale(1); }
                    33%      { transform: rotate(4deg) scale(1.02); }
                    66%      { transform: rotate(-3deg) scale(0.98); }
                }

                /* ── Subtle dot grid overlay ── */
                .dot-grid {
                    position: absolute;
                    inset: 0;
                    background-image: radial-gradient(rgba(99,102,241,0.08) 1px, transparent 1px);
                    background-size: 32px 32px;
                    z-index: 1;
                }

                /* ── Floating rings ── */
                .pulse-ring {
                    position: absolute;
                    border-radius: 50%;
                    border: 1px solid;
                    opacity: 0;
                    animation: ringPulse var(--dur) ease-out infinite;
                    animation-delay: var(--delay);
                    pointer-events: none;
                    z-index: 1;
                }
                @keyframes ringPulse {
                    0%   { transform: scale(0.5); opacity: 0.6; }
                    100% { transform: scale(2.5); opacity: 0; }
                }

                /* ── Left branding panel ── */
                .login-left {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    position: relative;
                    z-index: 2;
                    padding: 60px;
                }
                @media (max-width: 900px) {
                    .login-left { display: none; }
                }

                /* ── Animated heart icon ── */
                .heart-container {
                    position: relative;
                    width: 180px;
                    height: 180px;
                    margin-bottom: 48px;
                }
                .heart-outer {
                    position: absolute;
                    inset: 0;
                    border-radius: 50%;
                    border: 2px solid rgba(99,102,241,0.25);
                    animation: heartSpin 20s linear infinite;
                }
                .heart-outer::before {
                    content: '';
                    position: absolute;
                    top: -4px;
                    left: 50%;
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #818cf8;
                    box-shadow: 0 0 12px #818cf8, 0 0 24px rgba(99,102,241,0.5);
                }
                @keyframes heartSpin {
                    from { transform: rotate(0deg); }
                    to   { transform: rotate(360deg); }
                }
                .heart-mid {
                    position: absolute;
                    inset: 20px;
                    border-radius: 50%;
                    border: 1.5px solid rgba(34,211,238,0.15);
                    animation: heartSpin 15s linear infinite reverse;
                }
                .heart-mid::before {
                    content: '';
                    position: absolute;
                    bottom: -4px;
                    left: 50%;
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                    background: #22d3ee;
                    box-shadow: 0 0 10px #22d3ee;
                }
                .heart-inner {
                    position: absolute;
                    inset: 44px;
                    border-radius: 50%;
                    background: rgba(99,102,241,0.08);
                    border: 1px solid rgba(99,102,241,0.25);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    animation: heartBeat 1.8s ease-in-out infinite;
                    box-shadow: 0 0 30px rgba(99,102,241,0.15) inset;
                }
                @keyframes heartBeat {
                    0%, 100% { transform: scale(1); }
                    15%      { transform: scale(1.12); }
                    30%      { transform: scale(1); }
                    45%      { transform: scale(1.06); }
                    60%      { transform: scale(1); }
                }

                /* ── ECG line across the heart icon ── */
                .ecg-line {
                    position: absolute;
                    top: 50%;
                    left: -20px;
                    right: -20px;
                    height: 40px;
                    transform: translateY(-50%);
                    overflow: hidden;
                }
                .ecg-line svg {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 220px;
                    height: 40px;
                }
                .ecg-line svg polyline {
                    fill: none;
                    stroke: #22d3ee;
                    stroke-width: 2.2;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                    stroke-dasharray: 300;
                    stroke-dashoffset: 300;
                    animation: ecgDraw 2.5s ease-in-out infinite;
                    filter: drop-shadow(0 0 6px rgba(34,211,238,0.6));
                }
                @keyframes ecgDraw {
                    0%   { stroke-dashoffset: 300; }
                    50%  { stroke-dashoffset: 0; }
                    100% { stroke-dashoffset: -300; }
                }

                /* Stats row */
                .stat-pill {
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 18px;
                    border-radius: 40px;
                    background: rgba(99,102,241,0.1);
                    border: 1px solid rgba(99,102,241,0.25);
                    color: #a5b4fc;
                    font-size: 13px;
                    font-weight: 600;
                    backdrop-filter: blur(8px);
                }
                .stat-pill .stat-num {
                    color: #fff;
                    font-weight: 800;
                    font-size: 15px;
                    font-family: 'JetBrains Mono', monospace;
                }

                /* ── Right login panel ── */
                .login-right {
                    width: 520px;
                    min-width: 420px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    z-index: 2;
                    padding: 40px;
                }
                @media (max-width: 900px) {
                    .login-right { width: 100%; min-width: 0; }
                }

                .login-card {
                    width: 100%;
                    max-width: 400px;
                    padding: 44px 36px;
                    border-radius: 24px;
                    background: rgba(8, 17, 31, 0.75);
                    border: 1px solid rgba(99,102,241,0.2);
                    backdrop-filter: blur(40px);
                    box-shadow:
                        0 32px 64px rgba(0,0,0,0.5),
                        inset 0 1px 0 rgba(255,255,255,0.05);
                }

                .login-card .logo-bar {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 8px;
                }
                .login-card .logo-icon {
                    width: 42px;
                    height: 42px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #4f46e5, #3b82f6);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 4px 16px rgba(99,102,241,0.3);
                }
                .login-card h1 {
                    font-size: 26px;
                    font-weight: 900;
                    color: #fff;
                    margin: 0;
                    letter-spacing: -0.5px;
                }
                .login-card h1 span {
                    color: #818cf8;
                }
                .login-card .subtitle {
                    color: #94a3b8;
                    font-size: 14px;
                    margin: 6px 0 32px;
                }

                .field-group {
                    margin-bottom: 20px;
                }
                .field-group label {
                    display: block;
                    color: #94a3b8;
                    font-size: 12.5px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.6px;
                    margin-bottom: 8px;
                }
                .field-group input {
                    width: 100%;
                    padding: 13px 16px;
                    border-radius: 12px;
                    border: 1.5px solid rgba(99,102,241,0.2);
                    background: rgba(255,255,255,0.03);
                    color: #f1f5f9;
                    font-size: 15px;
                    font-family: inherit;
                    outline: none;
                    transition: border-color 0.25s, box-shadow 0.25s, background 0.25s;
                    box-sizing: border-box;
                }
                .field-group input::placeholder {
                    color: #475569;
                }
                .field-group input:focus {
                    border-color: rgba(99,102,241,0.6);
                    box-shadow: 0 0 0 4px rgba(99,102,241,0.1);
                    background: rgba(99,102,241,0.05);
                }

                .login-btn {
                    width: 100%;
                    padding: 14px;
                    border-radius: 12px;
                    border: none;
                    background: linear-gradient(135deg, #4f46e5, #3b82f6);
                    color: #fff;
                    font-size: 15px;
                    font-weight: 700;
                    font-family: inherit;
                    cursor: pointer;
                    margin-top: 8px;
                    transition: transform 0.2s, box-shadow 0.2s;
                    box-shadow: 0 4px 20px rgba(79,70,229,0.4);
                    position: relative;
                    overflow: hidden;
                }
                .login-btn:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 30px rgba(79,70,229,0.5);
                }
                .login-btn:disabled {
                    opacity: 0.7;
                    cursor: not-allowed;
                }
                .login-btn .btn-shine {
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                    animation: btnShine 3s ease-in-out infinite;
                }
                @keyframes btnShine {
                    0%   { left: -100%; }
                    50%  { left: 100%; }
                    100% { left: 100%; }
                }

                .error-msg {
                    text-align: center;
                    color: #f87171;
                    font-size: 13px;
                    font-weight: 600;
                    margin-top: 16px;
                    padding: 10px 14px;
                    border-radius: 10px;
                    background: rgba(239,68,68,0.1);
                    border: 1px solid rgba(239,68,68,0.25);
                }

                .login-footer {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    margin-top: 28px;
                    color: #64748b;
                    font-size: 11.5px;
                }

                /* ── Corner decorative lines ── */
                .corner-lines {
                    position: absolute;
                    width: 120px;
                    height: 120px;
                    z-index: 1;
                    pointer-events: none;
                }
                .corner-lines.tl { top: 30px; left: 30px; }
                .corner-lines.br { bottom: 30px; right: 30px; transform: rotate(180deg); }
                .corner-lines::before, .corner-lines::after {
                    content: '';
                    position: absolute;
                    background: rgba(99,102,241,0.25);
                }
                .corner-lines::before {
                    top: 0; left: 0;
                    width: 60px; height: 1.5px;
                }
                .corner-lines::after {
                    top: 0; left: 0;
                    width: 1.5px; height: 60px;
                }

                /* ── Spinner ── */
                .spin-loader {
                    width: 18px;
                    height: 18px;
                    border: 2.5px solid rgba(255,255,255,0.3);
                    border-top-color: #fff;
                    border-radius: 50%;
                    animation: spin 0.7s linear infinite;
                    display: inline-block;
                }
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }

                /* ── Float animation for particles ── */
                .float-particle {
                    position: absolute;
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 1;
                    animation: particleFloat var(--dur) ease-in-out infinite alternate;
                    animation-delay: var(--delay);
                }
                @keyframes particleFloat {
                    from { transform: translateY(0) translateX(0); opacity: var(--opa); }
                    to   { transform: translateY(var(--dy)) translateX(var(--dx)); opacity: calc(var(--opa) * 0.5); }
                }
            `}</style>

            <div className="login-page">
                {/* ── Background layers ── */}
                <div className="login-bg" />
                <div className="dot-grid" />

                {/* ── Corner decorative lines ── */}
                <div className="corner-lines tl" />
                <div className="corner-lines br" />

                {/* ── Pulse rings (center-left area) ── */}
                {[
                    { size: 200, x: "25%", y: "50%", color: "rgba(99,102,241,0.2)", dur: "4s", delay: "0s" },
                    { size: 300, x: "25%", y: "50%", color: "rgba(99,102,241,0.12)", dur: "4s", delay: "1s" },
                    { size: 400, x: "25%", y: "50%", color: "rgba(99,102,241,0.06)", dur: "4s", delay: "2s" },
                ].map((r, i) => (
                    <div key={i} className="pulse-ring" style={{
                        width: r.size, height: r.size,
                        left: `calc(${r.x} - ${r.size / 2}px)`,
                        top: `calc(${r.y} - ${r.size / 2}px)`,
                        borderColor: r.color,
                        "--dur": r.dur, "--delay": r.delay,
                    }} />
                ))}

                {/* ── Floating particles ── */}
                {[
                    { size: 4, x: "15%", y: "20%", dur: "6s", delay: "0s", dx: "15px", dy: "-20px", opa: 0.6, color: "#818cf8" },
                    { size: 3, x: "35%", y: "75%", dur: "8s", delay: "1s", dx: "-10px", dy: "-15px", opa: 0.5, color: "#22d3ee" },
                    { size: 5, x: "70%", y: "30%", dur: "7s", delay: "2s", dx: "12px", dy: "18px", opa: 0.45, color: "#818cf8" },
                    { size: 3, x: "80%", y: "70%", dur: "9s", delay: "0.5s", dx: "-8px", dy: "-12px", opa: 0.55, color: "#38bdf8" },
                    { size: 4, x: "50%", y: "15%", dur: "7s", delay: "1.5s", dx: "10px", dy: "14px", opa: 0.5, color: "#22d3ee" },
                    { size: 2, x: "10%", y: "60%", dur: "10s", delay: "3s", dx: "20px", dy: "-10px", opa: 0.6, color: "#818cf8" },
                    { size: 3, x: "90%", y: "45%", dur: "6s", delay: "2s", dx: "-14px", dy: "10px", opa: 0.4, color: "#38bdf8" },
                ].map((p, i) => (
                    <div key={i} className="float-particle" style={{
                        width: p.size, height: p.size,
                        left: p.x, top: p.y,
                        background: p.color,
                        boxShadow: `0 0 ${p.size * 3}px ${p.color}`,
                        "--dur": p.dur, "--delay": p.delay,
                        "--dx": p.dx, "--dy": p.dy, "--opa": p.opa,
                    }} />
                ))}

                {/* ══════════ Left Branding Panel ══════════ */}
                <div className="login-left">

                    {/* Animated heart icon */}
                    <div className="heart-container">
                        <div className="heart-outer" />
                        <div className="heart-mid" />
                        <div className="heart-inner">
                            <svg xmlns="http://www.w3.org/2000/svg" width="42" height="42" viewBox="0 0 24 24"
                                fill="none" stroke="#818cf8" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                            </svg>
                        </div>
                        {/* ECG line across heart */}
                        <div className="ecg-line">
                            <svg viewBox="0 0 220 40">
                                <polyline points="0,20 30,20 40,20 50,18 55,22 60,18 65,20 75,20 80,20 85,34 90,4 95,36 100,20 110,20 118,16 122,14 126,16 130,20 160,20 180,20 200,20 220,20" />
                            </svg>
                        </div>
                    </div>

                    {/* Title */}
                    <h2 style={{
                        fontSize: "42px", fontWeight: "900", color: "#fff",
                        letterSpacing: "-1.5px", margin: "0 0 12px",
                        textAlign: "center", lineHeight: 1.2,
                    }}>
                        Cardio<span style={{ color: "#818cf8" }}>AI</span>
                    </h2>
                    <p style={{
                        color: "#94a3b8", fontSize: "14px",
                        textAlign: "center", maxWidth: "380px",
                        lineHeight: 1.6, marginBottom: "36px",
                    }}>
                        A Triple-Teacher Knowledge Distillation Framework for Enhanced Multi-Disease Chest X-ray Classification.
                    </p>

                    {/* Stats row */}
                    <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", justifyContent: "center" }}>
                        <div className="stat-pill">
                            <span className="stat-num">93.79%</span> Macro-AUC
                        </div>
                        <div className="stat-pill">
                            <span className="stat-num">14</span> Diseases
                        </div>
                        <div className="stat-pill">
                            <span className="stat-num">112K+</span> X-rays
                        </div>
                    </div>
                </div>

                {/* ══════════ Right Login Card ══════════ */}
                <div className="login-right">
                    <div className="login-card">

                        {/* Logo bar */}
                        <div className="logo-bar">
                            <div className="logo-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
                                    fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                                </svg>
                            </div>
                            <h1>Cardio<span>AI</span></h1>
                        </div>
                        <p className="subtitle">Clinical Diagnostic Dashboard</p>

                        <form onSubmit={handleLogin}>
                            <div className="field-group">
                                <label>Username</label>
                                <input
                                    type="text"
                                    placeholder="Enter your username"
                                    value={username}
                                    onChange={e => setUsername(e.target.value)}
                                    autoComplete="username"
                                    required
                                />
                            </div>

                            <div className="field-group">
                                <label>Password</label>
                                <input
                                    type="password"
                                    placeholder="Enter your password"
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    autoComplete="current-password"
                                    required
                                />
                            </div>

                            <button type="submit" className="login-btn" disabled={loading || !username || !password}>
                                <span className="btn-shine" />
                                {loading ? (
                                    <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "10px" }}>
                                        <span className="spin-loader" /> Authenticating...
                                    </span>
                                ) : (
                                    "Sign In →"
                                )}
                            </button>
                        </form>

                        {error && <div className="error-msg">⚠️ {error}</div>}

                        <div className="login-footer">
                            <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
                                fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                            </svg>
                            HIPAA-compliant · End-to-end encrypted
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}

export default Login;
