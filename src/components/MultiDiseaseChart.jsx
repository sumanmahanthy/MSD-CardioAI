import React from "react";
import { Bar } from "react-chartjs-2";

import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// Co-morbidity threshold
const THRESHOLD = 45; // in %

function MultiDiseaseChart({ data }) {
    if (!data) return null;

    const labels = Object.keys(data);
    const values = Object.values(data).map(v => Math.round(v * 1000) / 10); // to %

    // Colour each bar: amber if above threshold (detected), else indigo
    const bgColors = values.map(v =>
        v >= THRESHOLD ? "rgba(245,158,11,0.85)" : "rgba(99,102,241,0.25)"
    );
    const borderColors = values.map(v =>
        v >= THRESHOLD ? "#f59e0b" : "#6366f1"
    );

    const chartData = {
        labels,
        datasets: [
            {
                label: "Disease Probability (%)",
                data: values,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 5,
                barThickness: 20,
                hoverBackgroundColor: values.map(v => v >= THRESHOLD ? "#f59e0b" : "#818cf8"),
            },
            // Threshold reference line
            {
                label: `Detection Threshold (${THRESHOLD}%)`,
                data: Array(labels.length).fill(THRESHOLD),
                type: "line",
                borderColor: "#f59e0b",
                borderWidth: 1.5,
                borderDash: [5, 4],
                pointRadius: 0,
                fill: false,
            }
        ]
    };

    const options = {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                display: true,
                position: "bottom",
                labels: { color: "#94a3b8", font: { family: "'Inter', sans-serif", size: 10 }, boxWidth: 12, padding: 16 }
            },
            tooltip: {
                backgroundColor: "rgba(8,17,31,0.95)",
                titleColor: "#e2e8f0",
                bodyColor: "#f8fafc",
                borderColor: "rgba(99,102,241,0.4)",
                borderWidth: 1,
                padding: 10,
                displayColors: false,
                callbacks: {
                    label: (ctx) => ` ${ctx.raw.toFixed(1)}%  ${ctx.raw >= THRESHOLD ? "⚠️ Detected" : "〰 Below threshold"}`,
                }
            }
        },
        scales: {
            x: {
                beginAtZero: true,
                max: 100,
                grid: { color: "rgba(255,255,255,0.03)" },
                border: { display: false },
                ticks: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 10 }, callback: v => v + "%" }
            },
            y: {
                grid: { display: false },
                border: { display: false },
                ticks: { color: "#94a3b8", font: { family: "'Inter', sans-serif", size: 11, weight: "600" } }
            }
        }
    };

    const detectedCount = values.filter(v => v >= THRESHOLD).length;

    return (
        <div className="card p-4 border-0 animate-fade-up">

            {/* ── Header ── */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "18px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{
                        width: "40px", height: "40px", borderRadius: "10px",
                        background: "rgba(99,102,241,0.1)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        border: "1px solid rgba(99,102,241,0.2)",
                    }}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                            fill="none" stroke="#818cf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
                        </svg>
                    </div>
                    <div>
                        <h6 style={{ margin: 0, fontWeight: 700, color: "var(--text-main)", fontSize: "14.5px" }}>
                            Secondary Findings
                        </h6>
                        <p style={{ margin: 0, fontSize: "11px", color: "var(--text-faint)" }}>
                            13 Pulmonary Co-morbidities
                        </p>
                    </div>
                </div>

                {/* Status Badge */}
                {detectedCount > 0 ? (
                    <span style={{
                        fontSize: "10px", padding: "4px 12px", borderRadius: "99px",
                        background: "rgba(245,158,11,0.12)", color: "#f59e0b",
                        border: "1px solid rgba(245,158,11,0.3)", fontWeight: 700,
                        letterSpacing: "0.5px", textTransform: "uppercase",
                    }}>
                        ⚠️ {detectedCount} Detected
                    </span>
                ) : (
                    <span style={{
                        fontSize: "10px", padding: "4px 12px", borderRadius: "99px",
                        background: "rgba(34,197,94,0.12)", color: "#4ade80",
                        border: "1px solid rgba(34,197,94,0.3)", fontWeight: 700,
                        letterSpacing: "0.5px", textTransform: "uppercase",
                    }}>
                        ✅ All Clear
                    </span>
                )}
            </div>

            {/* ── Chart Viewport ── */}
            <div style={{ width: "100%", height: "340px", position: "relative" }}>
                <Bar data={chartData} options={options} />
            </div>

        </div>
    );
}

export default MultiDiseaseChart;