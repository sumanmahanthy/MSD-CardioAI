import React from "react";

/**
 * ModelArchitecture — Triple-Teacher Knowledge Distillation Pipeline
 * ---------------------------------------------------------------
 * Visualises the CardioAI proposed architecture as a static
 * block diagram. No external library dependencies — pure CSS/SVG.
 *
 * Pipeline stages match the M.Tech thesis block diagram exactly:
 *   Block 1–4  : Preprocessing (CLAHE, Otsu Masking, Normalisation)
 *   Block 5    : Triple-Teacher Ensemble (ConvNeXt-V2-Base + Swin-Base + BioMedCLIP)
 *   Block 6    : Knowledge Distillation — Student (ConvNeXt-V2-Tiny)
 *   Block 7    : 3-Way Classification + Grad-CAM XAI
 */

const BOX = ({ color, title, sub, icon }) => (
    <div style={{
        background: color + '18',
        border: `1px solid ${color}44`,
        borderRadius: '10px',
        padding: '10px 14px',
        textAlign: 'center',
        minWidth: '0',
    }}>
        <div style={{ fontSize: '20px', marginBottom: '4px' }}>{icon}</div>
        <div style={{ fontSize: '12px', fontWeight: '700', color: color }}>{title}</div>
        {sub && <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px', lineHeight: '1.4' }}>{sub}</div>}
    </div>
);

const ARROW = ({ label }) => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '0 4px' }}>
        <div style={{ fontSize: '18px', color: '#64748b' }}>↓</div>
        {label && <div style={{ fontSize: '9px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{label}</div>}
    </div>
);

function ModelArchitecture() {
    return (
        <div className="card p-4 shadow-sm border-0">

            {/* Header */}
            <div className="d-flex align-items-center mb-1">
                <div className="me-2 d-flex align-items-center justify-content-center"
                    style={{ width:'36px', height:'36px', borderRadius:'8px',
                             background:'rgba(99,102,241,0.12)', color:'#6366f1' }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                         fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="2" y="3" width="20" height="14" rx="2"/>
                        <line x1="8" y1="21" x2="16" y2="21"/>
                        <line x1="12" y1="17" x2="12" y2="21"/>
                    </svg>
                </div>
                <div>
                    <h6 className="m-0 fw-bold text-main">Proposed Architecture</h6>
                    <p className="mb-0" style={{ fontSize: '10px', color: '#64748b' }}>
                        Triple-Teacher Knowledge Distillation → ConvNeXt-V2-Tiny Student
                    </p>
                </div>
            </div>

            <hr style={{ borderColor: 'rgba(148,163,184,0.1)', margin: '12px 0' }} />

            {/* ── Stage 1: Input ─────────────────────────────────────── */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0' }}>

                <BOX color="#6366f1" icon="🩻" title="Input Chest X-ray" sub="PA / AP frontal view (PNG / JPEG)" />
                <ARROW label="validate" />

                {/* ── Stage 2: Preprocessing ─────────────────────────── */}
                <div style={{ width: '100%', background: 'rgba(99,102,241,0.06)', border:'1px solid rgba(99,102,241,0.15)', borderRadius:'10px', padding:'10px 14px', marginBottom:'0' }}>
                    <div style={{ fontSize:'10px', fontWeight:'700', color:'#6366f1', marginBottom:'8px', textAlign:'center', textTransform:'uppercase', letterSpacing:'0.5px' }}>
                        Block 1–4 : Preprocessing Pipeline
                    </div>
                    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'8px' }}>
                        <BOX color="#8b5cf6" icon="⚙️" title="CLAHE Enhancement" sub="clipLimit=2.0, tile=8×8" />
                        <BOX color="#8b5cf6" icon="🫁" title="Otsu Lung Masking" sub="Morphological close (5×5)" />
                        <BOX color="#8b5cf6" icon="📐" title="Resize 384×384" sub="Bilinear interpolation" />
                        <BOX color="#8b5cf6" icon="📊" title="ImageNet Normalization" sub="μ=[0.485,0.456,0.406]" />
                    </div>
                </div>
                <ARROW label="tensor (1,3,384,384)" />

                {/* ── Stage 3: Triple-Teacher ──────────────────────────── */}
                <div style={{ width: '100%', background: 'rgba(245,158,11,0.06)', border:'1px solid rgba(245,158,11,0.2)', borderRadius:'10px', padding:'10px 14px' }}>
                    <div style={{ fontSize:'10px', fontWeight:'700', color:'#f59e0b', marginBottom:'8px', textAlign:'center', textTransform:'uppercase', letterSpacing:'0.5px' }}>
                        Block 5 : Triple-Teacher Ensemble (Soft-Label Generation)
                    </div>
                    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'6px' }}>
                        <BOX color="#f59e0b" icon="🧠" title="Teacher 1" sub="ConvNeXt-Base 384²" />
                        <BOX color="#f59e0b" icon="🧬" title="Teacher 2" sub="Swin-Base 384²" />
                        <BOX color="#f59e0b" icon="🔬" title="Teacher 3" sub="BioMedCLIP 224²" />
                    </div>
                    <div style={{ textAlign:'center', fontSize:'10px', color:'#94a3b8', marginTop:'8px' }}>
                        Ensemble fusion: 45% ConvNeXt + 45% Swin + 10% BioMedCLIP&nbsp;→&nbsp;Soft pseudo-labels (14 classes)
                    </div>
                </div>
                <ARROW label="KDLoss α=0.7" />

                {/* ── Stage 4: Student ───────────────────────────────── */}
                <BOX color="#22c55e" icon="🎓" title="Student: ConvNeXt-V2-Tiny" sub="27.9M params | Multi-Sample Dropout (×5) | 384×384" />
                <ARROW label="sigmoid (14 outputs)" />

                {/* ── Stage 5: 3-Way Classification ──────────────────── */}
                <div style={{ width: '100%', background: 'rgba(34,197,94,0.06)', border:'1px solid rgba(34,197,94,0.2)', borderRadius:'10px', padding:'10px 14px' }}>
                    <div style={{ fontSize:'10px', fontWeight:'700', color:'#22c55e', marginBottom:'8px', textAlign:'center', textTransform:'uppercase', letterSpacing:'0.5px' }}>
                        Block 6 : 3-Way Classification Output
                    </div>
                    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'8px' }}>
                        <BOX color="#ef4444" icon="🔴" title="Cardiomegaly" sub="prob ≥ 0.50" />
                        <BOX color="#f59e0b" icon="🟡" title="Co-morbidity" sub="other ≥ 0.45" />
                        <BOX color="#22c55e" icon="🟢" title="No Finding" sub="all < 0.45" />
                    </div>
                </div>
                <ARROW label="XAI" />

                {/* ── Stage 6: Grad-CAM ─────────────────────────────── */}
                <BOX color="#6366f1" icon="🗺️" title="Grad-CAM Heatmap" sub="Target: backbone.stages[-1].blocks[-1].norm | JET overlay" />

            </div>

            {/* Footer metric */}
            <div className="mt-3 p-2 text-center rounded-3" style={{ background:'rgba(34,197,94,0.08)', border:'1px solid rgba(34,197,94,0.2)' }}>
                <span style={{ fontSize:'11px', color:'#22c55e', fontWeight:'700' }}>
                    📈 Macro-AUC: 0.9379 &nbsp;|&nbsp; NIH ChestX-ray14 &nbsp;|&nbsp; 112,120 images &nbsp;|&nbsp; 14 disease classes
                </span>
            </div>
        </div>
    );
}

export default ModelArchitecture;