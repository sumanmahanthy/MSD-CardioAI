import React, { useState, useEffect, useRef } from "react";
import { getResponse } from "../chatbot/knowledge_base";

/* ── Markdown-lite renderer (bold + links + line breaks) ──────────────────── */
function MsgText({ text, isUser }) {
    return (
        <div>
            {text.split("\n").map((line, i) => {
                const parts = line.split(/\*\*(.*?)\*\*/g);
                return (
                    <div key={i} style={{ lineHeight: 1.6, minHeight: line === "" ? "6px" : undefined }}>
                        {parts.map((p, j) => {
                            if (j % 2 === 1) return <strong key={j}>{p}</strong>;
                            const segs = p.split(/(\[.*?\]\(.*?\))/g);
                            return segs.map((seg, k) => {
                                const m = seg.match(/\[(.*?)\]\((.*?)\)/);
                                return m
                                    ? <a key={k} href={m[2]} target="_blank" rel="noreferrer"
                                        style={{ color: isUser ? "rgba(255,255,255,0.9)" : "#059669", textDecoration: "underline" }}>{m[1]}</a>
                                    : seg;
                            });
                        })}
                    </div>
                );
            })}
        </div>
    );
}

/* ── Typing dots ──────────────────────────────────────────────────────────── */
function TypingDots() {
    return (
        <div style={{ display: "flex", gap: "5px", padding: "3px 2px" }}>
            {[0, 1, 2].map(i => (
                <span key={i} style={{
                    width: "7px", height: "7px", borderRadius: "50%",
                    background: "#10b981",
                    display: "inline-block",
                    animation: "cbTyping 1.2s ease-in-out infinite",
                    animationDelay: `${i * 0.2}s`,
                }} />
            ))}
        </div>
    );
}

/* ── Main ChatBot Component ───────────────────────────────────────────────── */
export default function ChatBot() {
    const [open, setOpen] = useState(false);
    const [msgs, setMsgs] = useState([]);
    const [input, setInput] = useState("");
    const [typing, setTyping] = useState(false);
    const [unread, setUnread] = useState(0);
    const [isDark, setIsDark] = useState(false);
    const bottomRef = useRef(null);
    const inputRef = useRef(null);

    /* Detect theme */
    useEffect(() => {
        const check = () => setIsDark(document.documentElement.getAttribute("data-theme") === "dark");
        check();
        const obs = new MutationObserver(check);
        obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
        return () => obs.disconnect();
    }, []);

    /* Greeting on first open */
    useEffect(() => {
        if (open && msgs.length === 0) {
            const greet = getResponse("hello");
            setMsgs([{ id: Date.now(), role: "bot", text: greet.response, quickReplies: greet.quickReplies }]);
        }
        if (open) { setUnread(0); setTimeout(() => inputRef.current?.focus(), 120); }
    }, [open]);

    /* Auto-scroll */
    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, typing]);

    const sendMessage = (text) => {
        const msg = text.trim();
        if (!msg) return;
        setMsgs(prev => [...prev, { id: Date.now(), role: "user", text: msg }]);
        setInput("");
        setTyping(true);
        setTimeout(() => {
            const res = getResponse(msg);
            setTyping(false);
            setMsgs(prev => [...prev, { id: Date.now() + 1, role: "bot", text: res.response, quickReplies: res.quickReplies }]);
            if (!open) setUnread(u => u + 1);
        }, 900 + Math.random() * 400);
    };

    const handleKey = (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
    };

    // ── Theme-aware colour tokens ──────────────────────────────────────────
    const t = isDark ? {
        windowBg: "#1e293b",
        windowBorder: "rgba(16,185,129,0.18)",
        headerBg: "linear-gradient(135deg,#064e3b,#065f46)",
        botBubbleBg: "rgba(255,255,255,0.07)",
        botBubbleBdr: "rgba(255,255,255,0.1)",
        botText: "#e2e8f0",
        inputBg: "rgba(255,255,255,0.06)",
        inputBdr: "rgba(16,185,129,0.25)",
        inputText: "#e2e8f0",
        inputPlch: "#475569",
        footerBg: "rgba(0,0,0,0.25)",
        footerBdr: "rgba(255,255,255,0.07)",
        qrBg: "rgba(16,185,129,0.1)",
        qrBdr: "rgba(16,185,129,0.3)",
        qrText: "#6ee7b7",
        qrHover: "rgba(16,185,129,0.2)",
        metaText: "#64748b",
    } : {
        windowBg: "#ffffff",
        windowBorder: "rgba(4,120,87,0.18)",
        headerBg: "linear-gradient(135deg,#047857,#059669)",
        botBubbleBg: "#f0fdf4",
        botBubbleBdr: "#d1fae5",
        botText: "#1e293b",
        inputBg: "#f8fafc",
        inputBdr: "rgba(4,120,87,0.3)",
        inputText: "#1e293b",
        inputPlch: "#94a3b8",
        footerBg: "#f8fafc",
        footerBdr: "#e2e8f0",
        qrBg: "rgba(4,120,87,0.07)",
        qrBdr: "rgba(4,120,87,0.25)",
        qrText: "#047857",
        qrHover: "rgba(4,120,87,0.15)",
        metaText: "#94a3b8",
    };

    return (
        <>
            <style>{`
                @keyframes cbSlideUp {
                    from { opacity:0; transform:translateY(16px) scale(0.97); }
                    to   { opacity:1; transform:translateY(0) scale(1); }
                }
                @keyframes cbTyping {
                    0%,60%,100% { transform:translateY(0); opacity:0.6; }
                    30%         { transform:translateY(-6px); opacity:1; }
                }
                @keyframes cbPulse {
                    0%   { box-shadow:0 6px 24px rgba(16,185,129,0.5),0 0 0 0 rgba(16,185,129,0.4); }
                    70%  { box-shadow:0 6px 24px rgba(16,185,129,0.3),0 0 0 14px rgba(16,185,129,0); }
                    100% { box-shadow:0 6px 24px rgba(16,185,129,0.5),0 0 0 0 rgba(16,185,129,0); }
                }
                .cb-scroll::-webkit-scrollbar { width:4px; }
                .cb-scroll::-webkit-scrollbar-track { background:transparent; }
                .cb-scroll::-webkit-scrollbar-thumb { background:rgba(16,185,129,0.25); border-radius:4px; }
                .cb-input { outline:none; }
                .cb-input::placeholder { color: var(--cb-plch); }
                .cb-input:focus { border-color: rgba(4,120,87,0.55) !important; box-shadow:0 0 0 3px rgba(4,120,87,0.1); }
            `}</style>

            {/* ── Chat Window ───────────────────────────────────────── */}
            {open && (
                <div style={{
                    position: "fixed",
                    bottom: "100px",
                    right: "28px",
                    zIndex: 9997,
                    width: "375px",
                    maxWidth: "calc(100vw - 36px)",
                    height: "530px",
                    borderRadius: "20px",
                    background: t.windowBg,
                    border: `1px solid ${t.windowBorder}`,
                    boxShadow: isDark
                        ? "0 24px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(16,185,129,0.06)"
                        : "0 20px 60px rgba(0,0,0,0.12), 0 0 0 1px rgba(4,120,87,0.06)",
                    display: "flex",
                    flexDirection: "column",
                    overflow: "hidden",
                    animation: "cbSlideUp 0.22s ease",
                }}>

                    {/* ── Header ──────────────────────────────────── */}
                    <div style={{
                        padding: "14px 16px",
                        background: t.headerBg,
                        display: "flex",
                        alignItems: "center",
                        gap: "12px",
                        flexShrink: 0,
                    }}>
                        {/* Bot avatar */}
                        <div style={{
                            width: "40px", height: "40px", borderRadius: "50%",
                            background: "rgba(255,255,255,0.15)",
                            border: "1.5px solid rgba(255,255,255,0.25)",
                            display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                        }}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                                fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                            </svg>
                        </div>
                        <div style={{ flex: 1 }}>
                            <div style={{ color: "#fff", fontWeight: "700", fontSize: "15px", letterSpacing: "-0.2px" }}>CardioBot</div>
                            <div style={{ color: "rgba(255,255,255,0.75)", fontSize: "11.5px", display: "flex", alignItems: "center", gap: "5px" }}>
                                <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "#6ee7b7", display: "inline-block" }} />
                                Online · Medical AI Assistant
                            </div>
                        </div>
                        <button onClick={() => setOpen(false)} style={{
                            background: "rgba(255,255,255,0.15)", border: "none", color: "#fff",
                            cursor: "pointer", fontSize: "16px", width: "30px", height: "30px",
                            borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
                            transition: "background 0.15s",
                        }}
                            onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.25)"}
                            onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.15)"}
                        >✕</button>
                    </div>

                    {/* ── Messages area ───────────────────────────── */}
                    <div className="cb-scroll" style={{
                        flex: 1, overflowY: "auto",
                        padding: "16px 14px",
                        display: "flex", flexDirection: "column", gap: "14px",
                        background: t.windowBg,
                    }}>
                        {msgs.map(m => (
                            <div key={m.id} style={{
                                display: "flex", flexDirection: "column",
                                alignItems: m.role === "user" ? "flex-end" : "flex-start",
                                gap: "6px",
                            }}>
                                {m.role === "bot" ? (
                                    <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", maxWidth: "90%" }}>
                                        {/* Small avatar */}
                                        <div style={{
                                            width: "28px", height: "28px", borderRadius: "50%",
                                            background: "linear-gradient(135deg,#047857,#10b981)",
                                            display: "flex", alignItems: "center", justifyContent: "center",
                                            flexShrink: 0, marginTop: "2px",
                                        }}>
                                            <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
                                                fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                                            </svg>
                                        </div>
                                        {/* Bot bubble */}
                                        <div style={{
                                            background: t.botBubbleBg,
                                            border: `1px solid ${t.botBubbleBdr}`,
                                            color: t.botText,
                                            borderRadius: "4px 16px 16px 16px",
                                            padding: "10px 14px",
                                            fontSize: "13px",
                                        }}>
                                            <MsgText text={m.text} isUser={false} />
                                        </div>
                                    </div>
                                ) : (
                                    /* User bubble */
                                    <div style={{
                                        background: "linear-gradient(135deg,#059669,#10b981)",
                                        color: "#fff",
                                        borderRadius: "16px 4px 16px 16px",
                                        padding: "10px 14px",
                                        maxWidth: "80%",
                                        fontSize: "13px",
                                    }}>
                                        <MsgText text={m.text} isUser={true} />
                                    </div>
                                )}

                                {/* Quick replies */}
                                {m.role === "bot" && m.quickReplies?.length > 0 && (
                                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", paddingLeft: "36px" }}>
                                        {m.quickReplies.map(qr => (
                                            <button key={qr} onClick={() => sendMessage(qr)} style={{
                                                padding: "5px 12px",
                                                borderRadius: "20px",
                                                border: `1px solid ${t.qrBdr}`,
                                                background: t.qrBg,
                                                color: t.qrText,
                                                fontSize: "11.5px",
                                                fontWeight: "600",
                                                cursor: "pointer",
                                                transition: "background 0.15s",
                                            }}
                                                onMouseEnter={e => e.currentTarget.style.background = t.qrHover}
                                                onMouseLeave={e => e.currentTarget.style.background = t.qrBg}
                                            >{qr}</button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}

                        {/* Typing indicator */}
                        {typing && (
                            <div style={{ display: "flex", alignItems: "flex-start", gap: "8px" }}>
                                <div style={{
                                    width: "28px", height: "28px", borderRadius: "50%",
                                    background: "linear-gradient(135deg,#047857,#10b981)",
                                    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                }}>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
                                        fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                                    </svg>
                                </div>
                                <div style={{
                                    background: t.botBubbleBg, border: `1px solid ${t.botBubbleBdr}`,
                                    borderRadius: "4px 16px 16px 16px", padding: "12px 16px",
                                }}>
                                    <TypingDots />
                                </div>
                            </div>
                        )}
                        <div ref={bottomRef} />
                    </div>

                    {/* ── Input bar ───────────────────────────────── */}
                    <div style={{
                        padding: "10px 12px",
                        borderTop: `1px solid ${t.footerBdr}`,
                        background: t.footerBg,
                        display: "flex",
                        gap: "8px",
                        alignItems: "center",
                        flexShrink: 0,
                    }}>
                        <textarea
                            ref={inputRef}
                            rows={1}
                            className="cb-input"
                            placeholder="Ask about cardiomegaly..."
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={handleKey}
                            style={{
                                flex: 1,
                                padding: "9px 14px",
                                borderRadius: "20px",
                                border: `1.5px solid ${t.inputBdr}`,
                                background: t.inputBg,
                                color: t.inputText,
                                fontSize: "13px",
                                outline: "none",
                                resize: "none",
                                maxHeight: "80px",
                                fontFamily: "inherit",
                                transition: "border-color 0.2s,box-shadow 0.2s",
                                "--cb-plch": t.inputPlch,
                            }}
                        />
                        <button
                            onClick={() => sendMessage(input)}
                            disabled={!input.trim()}
                            style={{
                                width: "38px",
                                height: "38px",
                                borderRadius: "50%",
                                border: "none",
                                background: input.trim()
                                    ? "linear-gradient(135deg,#059669,#10b981)"
                                    : (isDark ? "rgba(255,255,255,0.08)" : "#e2e8f0"),
                                color: input.trim() ? "#fff" : (isDark ? "#475569" : "#94a3b8"),
                                cursor: input.trim() ? "pointer" : "not-allowed",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                flexShrink: 0,
                                transition: "background 0.2s",
                                boxShadow: input.trim() ? "0 2px 10px rgba(16,185,129,0.35)" : "none",
                            }}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
                                fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13" />
                                <polygon points="22 2 15 22 11 13 2 9 22 2" />
                            </svg>
                        </button>
                    </div>

                    {/* Branding footer */}
                    <div style={{
                        textAlign: "center", padding: "6px", fontSize: "10.5px",
                        color: t.metaText, borderTop: `1px solid ${t.footerBdr}`,
                        background: t.footerBg, flexShrink: 0,
                    }}>
                        🔒 CardioBot · Medical AI Assistant · Not a substitute for clinical advice
                    </div>
                </div>
            )}

            {/* ── Floating Button ───────────────────────────────────── */}
            <button
                onClick={() => setOpen(o => !o)}
                title="Open CardioBot"
                style={{
                    position: "fixed",
                    bottom: "28px",
                    right: "28px",
                    zIndex: 9998,
                    width: "58px",
                    height: "58px",
                    borderRadius: "50%",
                    background: "linear-gradient(135deg,#059669,#10b981)",
                    border: "none",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "transform 0.2s",
                    animation: !open ? "cbPulse 2.2s infinite" : "none",
                    transform: open ? "scale(0.9) rotate(90deg)" : "scale(1) rotate(0deg)",
                    boxShadow: "0 6px 24px rgba(16,185,129,0.5)",
                }}
            >
                {open ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
                        fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
                        fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                    </svg>
                )}
                {/* Unread badge */}
                {!open && unread > 0 && (
                    <span style={{
                        position: "absolute", top: "-4px", right: "-4px",
                        background: "#ef4444", color: "#fff",
                        fontSize: "10px", fontWeight: "700",
                        width: "18px", height: "18px", borderRadius: "50%",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        border: `2px solid ${isDark ? "#0f172a" : "#ffffff"}`,
                    }}>
                        {unread}
                    </span>
                )}
            </button>
        </>
    );
}
