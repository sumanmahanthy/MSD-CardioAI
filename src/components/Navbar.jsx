import React, { useState, useEffect, useRef } from "react";

function Navbar({ onLogout }) {

    /* ── Theme ────────────────────────────────────────────────────────── */
    const [isDark, setIsDark] = useState(true); // default dark

    useEffect(() => {
        const saved = localStorage.getItem("theme");
        if (saved === "light") {
            setIsDark(false);
            document.documentElement.setAttribute("data-theme", "light");
        } else {
            // Default dark
            setIsDark(true);
            document.documentElement.removeAttribute("data-theme");
        }
    }, []);

    const toggleTheme = () => {
        const next = !isDark;
        setIsDark(next);
        if (!next) {
            document.documentElement.setAttribute("data-theme", "light");
            localStorage.setItem("theme", "light");
        } else {
            document.documentElement.removeAttribute("data-theme");
            localStorage.setItem("theme", "dark");
        }
    };

    /* ── Real-time Clock ──────────────────────────────────────────────── */
    const [time, setTime] = useState("");
    useEffect(() => {
        const fmt = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        setTime(fmt());
        const id = setInterval(() => setTime(fmt()), 1000);
        return () => clearInterval(id);
    }, []);

    /* ── Notifications ────────────────────────────────────────────────── */
    const [notifications, setNotifications] = useState([]);
    const [showNotif, setShowNotif]         = useState(false);
    const notifRef = useRef(null);

    const loadNotifications = () => {
        const stored = JSON.parse(localStorage.getItem("cardio_notifications") || "[]");
        setNotifications(stored);
    };

    useEffect(() => {
        loadNotifications();
        window.addEventListener("cardio_notify", loadNotifications);
        return () => window.removeEventListener("cardio_notify", loadNotifications);
    }, []);

    const unreadCount = notifications.filter(n => !n.read).length;

    const markAllRead = () => {
        const updated = notifications.map(n => ({ ...n, read: true }));
        setNotifications(updated);
        localStorage.setItem("cardio_notifications", JSON.stringify(updated));
    };

    const clearAll = () => {
        setNotifications([]);
        localStorage.removeItem("cardio_notifications");
    };

    const handleNotifOpen = () => {
        setShowNotif(v => !v);
        setShowProfile(false);
        if (!showNotif) markAllRead();
    };

    /* ── Profile ──────────────────────────────────────────────────────── */
    const [showProfile, setShowProfile] = useState(false);
    const profileRef = useRef(null);

    const handleProfileOpen = () => {
        setShowProfile(v => !v);
        setShowNotif(false);
    };

    /* ── Outside click ────────────────────────────────────────────────── */
    useEffect(() => {
        const handler = (e) => {
            if (notifRef.current  && !notifRef.current.contains(e.target))  setShowNotif(false);
            if (profileRef.current && !profileRef.current.contains(e.target)) setShowProfile(false);
        };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, []);

    /* ── Helpers ──────────────────────────────────────────────────────── */
    const labelColor = (lbl) => lbl === "Cardiomegaly" ? "#f87171" : (lbl === "No Finding" || lbl === "Normal") ? "#4ade80" : "#fbbf24";
    const labelBg    = (lbl) => lbl === "Cardiomegaly" ? "rgba(239,68,68,0.1)" : (lbl === "No Finding" || lbl === "Normal") ? "rgba(34,197,94,0.1)" : "rgba(245,158,11,0.1)";
    const labelIcon  = (lbl) => lbl === "Cardiomegaly" ? "🔴" : (lbl === "No Finding" || lbl === "Normal") ? "🟢" : "🟡";

    const dropdownBase = {
        position: "absolute",
        top: "calc(100% + 12px)",
        right: 0,
        background: "var(--card-bg)",
        border: "1px solid var(--border-color)",
        borderRadius: "18px",
        boxShadow: "var(--shadow-lg), 0 0 0 1px rgba(99,102,241,0.08)",
        zIndex: 9999,
        overflow: "hidden",
        animation: "dropIn 0.2s cubic-bezier(0.4,0,0.2,1)",
    };

    /* ── Render ───────────────────────────────────────────────────────── */
    return (
        <>
            <div className="navbar-dashboard">

                {/* ── Left: Status ── */}
                <div className="navbar-left">
                    <h5>System Status</h5>
                    <p>
                        <span className="status-dot online" />
                        AI Pipeline Ready · ConvNeXt-V2-Tiny Student KD
                    </p>
                </div>

                {/* ── Right: Controls ── */}
                <div className="navbar-right">

                    {/* Live status badge */}
                    <div className="navbar-status-badge" style={{ display: 'flex' }}>
                        <span className="live-dot" />
                        Backend Live
                    </div>

                    {/* Real-time clock */}
                    <div className="navbar-clock">{time}</div>

                    {/* Theme Toggle */}
                    <div
                        className="nav-icon"
                        title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
                        onClick={toggleTheme}
                    >
                        {isDark ? (
                            <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24"
                                fill="none" stroke="currentColor" strokeWidth="2"
                                strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="5"/>
                                <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                                <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                            </svg>
                        ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24"
                                fill="none" stroke="currentColor" strokeWidth="2"
                                strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                            </svg>
                        )}
                    </div>

                    {/* ── Notifications Bell ── */}
                    <div ref={notifRef} style={{ position: "relative" }}>
                        <div
                            className="nav-icon"
                            title="Notifications"
                            onClick={handleNotifOpen}
                            style={{ position: "relative" }}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" fill="none"
                                stroke="currentColor" strokeWidth="2"
                                strokeLinecap="round" strokeLinejoin="round">
                                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                                <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
                            </svg>

                            {unreadCount > 0 && (
                                <span style={{
                                    position: "absolute",
                                    top: "-5px", right: "-5px",
                                    background: "#ef4444",
                                    color: "#fff",
                                    fontSize: "9px", fontWeight: 800,
                                    width: "16px", height: "16px",
                                    borderRadius: "50%",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    border: "2px solid var(--card-bg)",
                                    boxShadow: "0 0 6px rgba(239,68,68,0.5)",
                                    animation: "livePulse 2s infinite",
                                }}>
                                    {unreadCount > 9 ? "9+" : unreadCount}
                                </span>
                            )}
                        </div>

                        {showNotif && (
                            <div style={{ ...dropdownBase, minWidth: "320px" }}>
                                {/* Header */}
                                <div style={{
                                    padding: "14px 18px 12px",
                                    borderBottom: "1px solid var(--border-subtle)",
                                    display: "flex", alignItems: "center", justifyContent: "space-between",
                                    background: "linear-gradient(135deg, rgba(99,102,241,0.05), transparent)",
                                }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                        <div style={{
                                            width: "28px", height: "28px", borderRadius: "8px",
                                            background: "rgba(99,102,241,0.12)",
                                            display: "flex", alignItems: "center", justifyContent: "center",
                                            fontSize: "14px",
                                        }}>🔔</div>
                                        <span style={{ fontWeight: 800, fontSize: "13px", color: "var(--text-main)" }}>
                                            AI Predictions
                                        </span>
                                        {notifications.length > 0 && (
                                            <span style={{
                                                padding: "2px 7px", borderRadius: "99px",
                                                background: "rgba(99,102,241,0.12)", color: "var(--secondary)",
                                                fontSize: "10px", fontWeight: 700,
                                            }}>
                                                {notifications.length}
                                            </span>
                                        )}
                                    </div>
                                    {notifications.length > 0 && (
                                        <button onClick={clearAll} style={{
                                            background: "none", border: "none",
                                            color: "var(--text-faint)", fontSize: "11px",
                                            cursor: "pointer", padding: "2px 8px",
                                            borderRadius: "6px", fontWeight: 600,
                                            transition: "color 0.2s",
                                        }}
                                            onMouseEnter={e => e.currentTarget.style.color = '#f87171'}
                                            onMouseLeave={e => e.currentTarget.style.color = 'var(--text-faint)'}
                                        >
                                            Clear all
                                        </button>
                                    )}
                                </div>

                                {/* List */}
                                <div style={{ maxHeight: "300px", overflowY: "auto" }}>
                                    {notifications.length === 0 ? (
                                        <div style={{
                                            padding: "40px 16px", textAlign: "center",
                                            color: "var(--text-faint)", fontSize: "13px",
                                        }}>
                                            <div style={{ fontSize: "28px", marginBottom: "10px", opacity: 0.5 }}>🔕</div>
                                            No predictions yet.<br />Upload an X-ray to start.
                                        </div>
                                    ) : (
                                        notifications.slice(0, 6).map(n => (
                                            <div key={n.id} style={{
                                                padding: "12px 16px",
                                                borderBottom: "1px solid var(--border-subtle)",
                                                display: "flex", alignItems: "center", gap: "12px",
                                                background: n.read ? "transparent" : "rgba(99,102,241,0.04)",
                                                transition: "background 0.2s",
                                                cursor: "default",
                                            }}
                                                onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.06)'}
                                                onMouseLeave={e => e.currentTarget.style.background = n.read ? 'transparent' : 'rgba(99,102,241,0.04)'}
                                            >
                                                <div style={{
                                                    width: "36px", height: "36px",
                                                    borderRadius: "10px",
                                                    background: labelBg(n.label),
                                                    display: "flex", alignItems: "center",
                                                    justifyContent: "center",
                                                    fontSize: "16px", flexShrink: 0,
                                                    border: `1px solid ${labelColor(n.label)}25`,
                                                }}>
                                                    {labelIcon(n.label)}
                                                </div>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ fontWeight: 700, fontSize: "12.5px", color: labelColor(n.label) }}>
                                                        {n.label}
                                                    </div>
                                                    <div style={{ fontSize: "11px", color: "var(--text-faint)" }}>
                                                        Confidence: {(n.confidence * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                                <div style={{ fontSize: "10px", color: "var(--text-faint)", whiteSpace: "nowrap", fontFamily: "'JetBrains Mono', monospace" }}>
                                                    {n.time}
                                                </div>
                                            </div>
                                        ))
                                    )}
                                </div>

                                {notifications.length > 6 && (
                                    <div style={{
                                        padding: "10px 16px", textAlign: "center",
                                        borderTop: "1px solid var(--border-subtle)",
                                        background: "rgba(99,102,241,0.02)",
                                    }}>
                                        <span style={{ fontSize: "12px", color: "var(--secondary)", fontWeight: 700 }}>
                                            +{notifications.length - 6} more in History
                                        </span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* ── Profile Avatar ── */}
                    <div ref={profileRef} style={{ position: "relative" }}>
                        <div
                            className="user-avatar"
                            title="Profile"
                            onClick={handleProfileOpen}
                            style={{ boxShadow: showProfile ? "var(--shadow-glow)" : undefined }}
                        >
                            DR
                        </div>

                        {showProfile && (
                            <div style={{ ...dropdownBase, minWidth: "260px" }}>
                                {/* Profile header */}
                                <div style={{
                                    padding: "24px 18px 18px",
                                    borderBottom: "1px solid var(--border-subtle)",
                                    textAlign: "center",
                                    background: "linear-gradient(135deg, rgba(99,102,241,0.07), transparent)",
                                    position: "relative",
                                    overflow: "hidden",
                                }}>
                                    {/* Background glow */}
                                    <div style={{
                                        position: "absolute", top: -30, left: "50%", transform: "translateX(-50%)",
                                        width: 120, height: 120, borderRadius: "50%",
                                        background: "radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%)",
                                        pointerEvents: "none",
                                    }} />
                                    <div style={{
                                        width: "58px", height: "58px", borderRadius: "16px",
                                        background: "linear-gradient(135deg, #4338ca, #6366f1, #7c3aed)",
                                        color: "#fff",
                                        display: "flex", alignItems: "center", justifyContent: "center",
                                        fontWeight: 900, fontSize: "19px",
                                        margin: "0 auto 12px",
                                        boxShadow: "0 6px 24px rgba(99,102,241,0.5)",
                                        position: "relative",
                                    }}>
                                        DR
                                    </div>
                                    <div style={{ fontWeight: 800, fontSize: "16px", color: "var(--text-main)" }}>
                                        Dr. Suman
                                    </div>
                                    <div style={{ fontSize: "11.5px", color: "var(--text-faint)", marginTop: "2px" }}>
                                        Chief Radiologist
                                    </div>
                                    <span style={{
                                        display: "inline-flex", alignItems: "center", gap: "5px",
                                        marginTop: "10px", padding: "4px 12px",
                                        borderRadius: "99px",
                                        background: "rgba(34,197,94,0.08)",
                                        color: "#4ade80",
                                        fontSize: "11px", fontWeight: 700,
                                        border: "1px solid rgba(34,197,94,0.2)",
                                    }}>
                                        <span className="live-dot" />
                                        Online
                                    </span>
                                </div>

                                {/* Info rows */}
                                <div style={{ padding: "6px 0" }}>
                                    {[
                                        { icon: "🏥", label: "Institution", value: "VIIT Medical Centre" },
                                        { icon: "📋", label: "Speciality",  value: "Cardiac Radiology" },
                                        { icon: "🎓", label: "Degree",      value: "M.Tech — AI in Healthcare" },
                                    ].map(row => (
                                        <div key={row.label} style={{
                                            padding: "10px 18px",
                                            display: "flex", alignItems: "flex-start", gap: "12px",
                                        }}>
                                            <span style={{ fontSize: "15px", flexShrink: 0, marginTop: "1px" }}>{row.icon}</span>
                                            <div>
                                                <div style={{ fontSize: "9.5px", color: "var(--text-faint)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.8px" }}>
                                                    {row.label}
                                                </div>
                                                <div style={{ fontSize: "12.5px", color: "var(--text-main)", fontWeight: 500, marginTop: "1px" }}>
                                                    {row.value}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Logout */}
                                <div style={{ padding: "10px 16px 14px", borderTop: "1px solid var(--border-subtle)" }}>
                                    <button
                                        onClick={() => { setShowProfile(false); onLogout(); }}
                                        style={{
                                            width: "100%", padding: "10px",
                                            borderRadius: "10px", border: "1px solid rgba(239,68,68,0.2)",
                                            background: "rgba(239,68,68,0.06)",
                                            color: "#f87171",
                                            fontWeight: 700, fontSize: "13px",
                                            cursor: "pointer",
                                            display: "flex", alignItems: "center",
                                            justifyContent: "center", gap: "8px",
                                            transition: "all 0.2s",
                                        }}
                                        onMouseEnter={e => {
                                            e.currentTarget.style.background = "rgba(239,68,68,0.12)";
                                            e.currentTarget.style.borderColor = "rgba(239,68,68,0.35)";
                                        }}
                                        onMouseLeave={e => {
                                            e.currentTarget.style.background = "rgba(239,68,68,0.06)";
                                            e.currentTarget.style.borderColor = "rgba(239,68,68,0.2)";
                                        }}
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none"
                                            stroke="currentColor" strokeWidth="2"
                                            strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                                            <polyline points="16 17 21 12 16 7"/>
                                            <line x1="21" y1="12" x2="9" y2="12"/>
                                        </svg>
                                        Sign Out
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                </div>
            </div>
        </>
    );
}

export default Navbar;