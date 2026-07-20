import React from "react";
import { NavLink } from "react-router-dom";

const navItems = [
    {
        to: "/",
        end: true,
        label: "Dashboard",
        icon: (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/>
                <rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>
            </svg>
        ),
    },
    {
        to: "/upload",
        label: "Analyse X-ray",
        icon: (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
        ),
    },
    {
        to: "/predictions",
        label: "Predictions",
        icon: (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
                <polyline points="10 9 9 9 8 9"/>
            </svg>
        ),
    },
    {
        to: "/analytics",
        label: "Analytics",
        icon: (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        ),
    },
    {
        to: "/history",
        label: "History",
        icon: (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
        ),
    },
];

function Sidebar() {
    return (
        <div className="sidebar">

            {/* ── Brand Header ── */}
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <div className="sidebar-logo-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
                            fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                        </svg>
                    </div>
                    <div>
                        <h4>CardioAI</h4>
                        <p>AI Radiology Platform</p>
                    </div>
                </div>
            </div>

            {/* ── Navigation ── */}
            <ul className="sidebar-menu">
                {navItems.map(item => (
                    <li key={item.to}>
                        <NavLink
                            to={item.to}
                            end={item.end}
                            className={({ isActive }) => isActive ? "active" : ""}
                        >
                            {item.icon}
                            {item.label}
                        </NavLink>
                    </li>
                ))}
            </ul>

            {/* ── Model Badge Footer ── */}
            <div className="sidebar-footer">
                <div className="sidebar-model-badge">
                    <span className="badge-label">Active Model</span>
                    <div className="badge-value">ConvNeXt-V2-Tiny</div>
                    <div className="badge-auc">
                        <span className="live-dot" />
                        AUC 93.79% · 14 Diseases
                    </div>
                </div>
            </div>

        </div>
    );
}

export default Sidebar;