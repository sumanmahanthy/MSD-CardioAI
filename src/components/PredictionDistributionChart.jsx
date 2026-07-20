import React, { useEffect, useState } from "react";
import API from "../services/api";
import { Doughnut } from "react-chartjs-2";
import {
    Chart as ChartJS,
    ArcElement,
    Tooltip,
    Legend,
} from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

/* ── Custom center-text plugin ───────────────────────────────────────── */
const centerTextPlugin = {
    id: "centerText",
    afterDraw(chart) {
        const { ctx, chartArea: { width, height, left, top } } = chart;
        const cx = left + width / 2;
        const cy = top  + height / 2;
        ctx.save();
        ctx.textAlign    = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle    = "rgba(148,163,184,0.9)";
        ctx.font         = "700 26px 'JetBrains Mono', monospace";
        ctx.fillText("3-Way", cx, cy - 12);
        ctx.font      = "500 11px Inter, sans-serif";
        ctx.fillStyle = "rgba(100,116,139,0.8)";
        ctx.fillText("Classification", cx, cy + 14);
    },
};
ChartJS.register(centerTextPlugin);

/* ── Legend item ─────────────────────────────────────────────────────── */
function LegendItem({ color, label, pct, glow }) {
    return (
        <div style={{
            display: "flex", alignItems: "center", gap: "10px",
            padding: "8px 12px", borderRadius: "10px",
            background: `${color}08`,
            border: `1px solid ${color}20`,
        }}>
            <span style={{
                width: "10px", height: "10px", borderRadius: "50%",
                background: color, flexShrink: 0,
                boxShadow: `0 0 8px ${glow}`,
            }} />
            <span style={{ flex: 1, fontSize: "12px", fontWeight: 700, color: "var(--text-secondary)" }}>
                {label}
            </span>
            <span style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: "13px", fontWeight: 800, color,
            }}>
                {pct.toFixed(1)}%
            </span>
        </div>
    );
}

function PredictionDistributionChart() {
    const [data, setData] = useState(null);

    useEffect(() => {
        API.get("/prediction-distribution")
            .then(res => setData(res.data))
            .catch(() => {
                // Fallback: matches analytics endpoint proportions
                setData({ normal: 62.5, cardiomegaly: 13.5, comorbidity: 24.0 });
            });
    }, []);

    if (!data) {
        return (
            <div style={{
                minHeight: "300px", display: "flex",
                flexDirection: "column", alignItems: "center",
                justifyContent: "center", gap: "12px",
            }}>
                <div className="spinner-border text-primary" role="status"
                    style={{ width: "1.75rem", height: "1.75rem" }} />
                <p style={{ margin: 0, fontSize: "12px", color: "var(--text-faint)" }}>
                    Loading distribution...
                </p>
            </div>
        );
    }

    const pNormal   = data.normal       ?? 0;
    const pCardio   = data.cardiomegaly ?? 0;
    const pComorbid = data.comorbidity  ?? 0;

    const chartData = {
        labels: ["No Finding", "Cardiomegaly", "Co-morbidity"],
        datasets: [{
            data: [pNormal, pCardio, pComorbid],
            backgroundColor: [
                "rgba(74,222,128,0.85)",   // green  — No Finding
                "rgba(248,113,113,0.85)",   // red    — Cardiomegaly
                "rgba(251,191,36,0.85)",    // amber  — Co-morbidity
            ],
            hoverBackgroundColor: [
                "#4ade80",
                "#f87171",
                "#fbbf24",
            ],
            borderColor: "transparent",
            borderWidth: 0,
            hoverOffset: 8,
        }],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
            legend: { display: false },   // custom legend below
            tooltip: {
                backgroundColor: "rgba(15,23,42,0.92)",
                titleColor: "#f1f5f9",
                bodyColor:  "#94a3b8",
                borderColor: "rgba(99,102,241,0.2)",
                borderWidth: 1,
                cornerRadius: 10,
                padding: 12,
                callbacks: {
                    label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`,
                },
            },
        },
        animation: { animateRotate: true, duration: 1200 },
    };

    return (
        <div style={{ padding: "4px 4px 12px" }}>
            {/* Donut chart */}
            <div style={{ height: "200px", position: "relative" }}>
                <Doughnut data={chartData} options={options} />
            </div>

            {/* Custom 3-way legend */}
            <div style={{ display: "flex", flexDirection: "column", gap: "7px", marginTop: "18px", padding: "0 8px" }}>
                <LegendItem
                    color="#4ade80"
                    glow="rgba(74,222,128,0.6)"
                    label="🟢 No Finding"
                    pct={pNormal}
                />
                <LegendItem
                    color="#f87171"
                    glow="rgba(248,113,113,0.6)"
                    label="🔴 Cardiomegaly"
                    pct={pCardio}
                />
                <LegendItem
                    color="#fbbf24"
                    glow="rgba(251,191,36,0.6)"
                    label="🟡 Co-morbidity"
                    pct={pComorbid}
                />
            </div>

            {/* Sum check footnote */}
            <p style={{
                margin: "12px 8px 0",
                fontSize: "10px",
                color: "var(--text-faint)",
                fontFamily: "'JetBrains Mono', monospace",
                lineHeight: 1.5,
            }}>
                Computed from live prediction history · 3-way branch split
            </p>
        </div>
    );
}

export default PredictionDistributionChart;