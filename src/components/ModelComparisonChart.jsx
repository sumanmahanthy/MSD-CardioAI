import React, { useState } from "react";
import { Bar, Radar } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale, LinearScale, BarElement,
    RadialLinearScale, PointElement, LineElement,
    Title, Tooltip, Legend, Filler
} from "chart.js";

ChartJS.register(
    CategoryScale, LinearScale, BarElement,
    RadialLinearScale, PointElement, LineElement,
    Title, Tooltip, Legend, Filler
);

// ── Paper data (Macro-AUC × 100, NIH ChestX-ray14 benchmark) ─────────────────
const papers = [
    { shortLabel: "✨ Proposed",        architecture: "CardioAI: 3-Teacher KD → ConvNeXt-V2-Tiny",          accuracy: 93.79, isProposed: true },
    { shortLabel: "P11: Hybrid CNN",    architecture: "Hybrid Convolutional + OOD Detection (VinBig/NIH)",      accuracy: 91.06 },
    { shortLabel: "P1: EfficientNet",  architecture: "EfficientNet-B0/B2",                                      accuracy: 87.30 },
    { shortLabel: "P2: CheXNet",       architecture: "DenseNet-121 (CheXNet)",                                  accuracy: 84.10 },
    { shortLabel: "P3: Swin Transformer", architecture: "Swin-Base Transformer",                                accuracy: 83.45 },
    { shortLabel: "P4: ResNet",        architecture: "ResNet18 + AlexNet Ensemble",                             accuracy: 82.50 },
    { shortLabel: "P5: ConvNeXt+DP",   architecture: "ConvNeXt with Drop Path",                                 accuracy: 84.85 },
    { shortLabel: "P6: BioMedCLIP",    architecture: "BioMedCLIP Fine-tuned",                                   accuracy: 83.45 },
    { shortLabel: "P7: MedFusion",     architecture: "CNN + Transformer Hybrid",                                 accuracy: 81.20 },
    { shortLabel: "P8: DenseNet",      architecture: "DenseNet-121 Baseline",                                    accuracy: 80.50 },
    { shortLabel: "P9: Swin-B",        architecture: "Swin-B (our Swin Teacher)",                               accuracy: 84.83 },
];

// ── Bar chart ─────────────────────────────────────────────────────────────────
const barData = {
    labels: papers.map(p => p.shortLabel),
    datasets: [{
        label: "Accuracy (%)",
        data: papers.map(p => p.accuracy),
        backgroundColor: papers.map(p =>
            p.isProposed ? "rgba(99,102,241,0.9)" : "rgba(100,116,139,0.45)"
        ),
        borderColor: papers.map(p =>
            p.isProposed ? "rgba(139,92,246,1)" : "rgba(100,116,139,0.7)"
        ),
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
    }],
};

const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { display: false },
        tooltip: {
            callbacks: {
                title: (ctx) => papers[ctx[0].dataIndex].shortLabel,
                label: (ctx) => [
                    `  Accuracy : ${ctx.raw}%`,
                    `  Model    : ${papers[ctx.dataIndex].architecture}`,
                ],
            },
            backgroundColor: "rgba(15,23,42,0.95)",
            titleColor: "#e2e8f0",
            bodyColor: "#94a3b8",
            padding: 12,
            cornerRadius: 10,
            borderWidth: 1,
            borderColor: "rgba(99,102,241,0.35)",
        },
    },
    scales: {
        y: {
            min: 78,
            max: 96,
            ticks: { color: "#64748b", callback: v => `${v}%`, stepSize: 5 },
            grid: { color: "rgba(255,255,255,0.05)" },
            border: { display: false },
        },
        x: {
            ticks: { color: "#94a3b8", maxRotation: 38, minRotation: 25, font: { size: 11 } },
            grid: { display: false },
            border: { display: false },
        },
    },
};

// ── Radar chart ───────────────────────────────────────────────────────────────
const radarData = {
    labels: ["Macro-AUC", "Multi-Teacher\nDistillation", "Knowledge\nDistillation", "3-View TTA", "Multi-Disease\nSupport", "Model\nCompression"],
    datasets: [
        {
            label: "CardioAI (Proposed)",
            data: [93.79, 100, 100, 100, 100, 90],
            backgroundColor: "rgba(99,102,241,0.22)",
            borderColor: "rgba(99,102,241,1)",
            pointBackgroundColor: "#6366f1",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            borderWidth: 2.5,
        },
        {
            label: "Best Existing Paper (P11 Hybrid CNN)",
            data: [91.06, 0, 0, 0, 70, 0],
            backgroundColor: "rgba(245,158,11,0.13)",
            borderColor: "rgba(245,158,11,0.8)",
            pointBackgroundColor: "#f59e0b",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            borderWidth: 2,
        },
    ],
};

const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
        r: {
            min: 0,
            max: 100,
            ticks: { color: "#64748b", backdropColor: "transparent", stepSize: 25 },
            grid: { color: "rgba(255,255,255,0.07)" },
            angleLines: { color: "rgba(255,255,255,0.07)" },
            pointLabels: { color: "#94a3b8", font: { size: 11, weight: "600" } },
        },
    },
    plugins: {
        legend: {
            position: "bottom",
            labels: { color: "#94a3b8", padding: 20, font: { size: 12 } },
        },
        tooltip: {
            backgroundColor: "rgba(15,23,42,0.95)",
            titleColor: "#e2e8f0",
            bodyColor: "#94a3b8",
            padding: 10,
            cornerRadius: 8,
            borderWidth: 1,
            borderColor: "rgba(99,102,241,0.3)",
        },
    },
};

// ── Component ─────────────────────────────────────────────────────────────────
export default function ModelComparisonChart() {
    const [activeTab, setActiveTab] = useState("bar");

    const tabBtn = (id, icon, label) => (
        <button
            onClick={() => setActiveTab(id)}
            style={{
                padding: "7px 20px",
                borderRadius: "20px",
                border: "none",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "13px",
                transition: "all 0.22s",
                background: activeTab === id
                    ? "linear-gradient(135deg,#6366f1,#8b5cf6)"
                    : "rgba(100,116,139,0.15)",
                color: activeTab === id ? "#fff" : "#94a3b8",
                boxShadow: activeTab === id ? "0 4px 14px rgba(99,102,241,0.4)" : "none",
            }}
        >
            {icon}&nbsp;{label}
        </button>
    );

    return (
        <div className="card border-0 shadow-sm" style={{
            background: "var(--card-bg, #0f172a)",
            borderRadius: "16px",
            padding: "24px",
        }}>
            {/* ── Header ─────────────────────────────────────────── */}
            <div className="d-flex justify-content-between align-items-start flex-wrap gap-3 mb-4">
                <div>
                    <h5 style={{ color: "var(--text-primary,#f8fafc)", fontWeight: "700", margin: 0 }}>
                        📊 Model Performance Comparison
                    </h5>
                    <p style={{ color: "#64748b", fontSize: "13px", margin: "4px 0 0" }}>
                        Your proposed model vs. 10 existing research papers
                    </p>
                </div>
                <div className="d-flex gap-2">
                    {tabBtn("bar", "📊", "Bar Chart")}
                    {tabBtn("radar", "🕸️", "Radar Chart")}
                </div>
            </div>

            {/* ── Champion Banner ─────────────────────────────────── */}
            <div style={{
                background: "linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.07))",
                border: "1px solid rgba(99,102,241,0.25)",
                borderRadius: "12px",
                padding: "14px 20px",
                marginBottom: "24px",
                display: "flex",
                alignItems: "center",
                gap: "14px",
                flexWrap: "wrap",
            }}>
                <span style={{ fontSize: "30px" }}>🏆</span>
                <div>
                    <div style={{ color: "#818cf8", fontWeight: "700", fontSize: "15px" }}>
                        Best Macro-AUC: 93.79% — CardioAI (Proposed) vs Paper11: 91.06%
                    </div>
                    <div style={{ color: "#64748b", fontSize: "12px", marginTop: "2px" }}>
                        3-Teacher Distillation (ConvNeXtV2-Base + Swin-Base + BioMedCLIP) → ConvNeXt-V2-Tiny Student · 9× faster inference · NIH ChestX-ray14 · 14 Diseases incl. Hernia
                    </div>
                </div>
            </div>

            {/* ── Bar Chart ──────────────────────────────────────── */}
            {activeTab === "bar" && (
                <div style={{ height: "370px" }}>
                    <Bar data={barData} options={barOptions} />
                </div>
            )}

            {/* ── Radar Chart ────────────────────────────────────── */}
            {activeTab === "radar" && (
                <div style={{ height: "390px" }}>
                    <Radar data={radarData} options={radarOptions} />
                </div>
            )}

            {/* ── Innovation Badges ──────────────────────────────── */}
            <div style={{
                marginTop: "22px",
                paddingTop: "18px",
                borderTop: "1px solid rgba(255,255,255,0.07)",
            }}>
                <p style={{ color: "#475569", fontSize: "11px", fontWeight: "600", letterSpacing: "1px", textTransform: "uppercase", marginBottom: "10px" }}>
                    Novel Contributions of Your Model
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                    {[
                        "🎓 3-Teacher Knowledge Distillation",
                        "🧠 ConvNeXt-V2 + Swin + BioMedCLIP",
                        "⚡ 9× Faster at Inference (28M params)",
                        "🔥 Multi-Sample Dropout Ensemble",
                        "📊 3-View Test-Time Augmentation",
                        "🏆 93.79% Macro-AUC on NIH 14",
                    ].map(b => (
                        <span key={b} style={{
                            padding: "5px 13px",
                            borderRadius: "20px",
                            background: "rgba(99,102,241,0.1)",
                            border: "1px solid rgba(99,102,241,0.22)",
                            color: "#818cf8",
                            fontSize: "12px",
                            fontWeight: "600",
                        }}>
                            {b}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}
