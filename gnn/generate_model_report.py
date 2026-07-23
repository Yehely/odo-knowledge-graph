"""
Generate a comprehensive HTML report visualising the OpioidGNN architecture,
training procedure, and data statistics.

Usage:
    conda run -n odo python3 -m gnn.generate_model_report
Outputs:
    gnn/model_report.html
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.config import (
    BATCH_SIZE, DROPOUT, GNN_LAYERS, LR, LR_PATIENCE,
    MAX_EPOCHS, MORGAN_BITS, MORGAN_RADIUS, PATIENCE,
    TEMPORAL_CUTOFF_YEAR, VAL_FRAC_OF_TRAIN, WEIGHT_DECAY,
)
from gnn.dataset import build_dataset
from gnn.model import build_model

# ───────────────────────────────────────────────────────────────────────────
# 1.  Collect exact numbers from the live dataset & model
# ───────────────────────────────────────────────────────────────────────────
print("Loading data and model …")
data, meta = build_dataset(verbose=False)
et = ("compound", "activity", "target")
edge_dim = data[et].edge_attr.shape[1]
model = build_model(data, edge_dim=edge_dim)

# Counts
N_c   = data["compound"].num_nodes
N_t   = data["target"].num_nodes
E     = int(data[et].edge_index.shape[1])
E_tr  = int(data[et].train_mask.sum())
E_val = int(data[et].val_mask.sum())
E_te  = int(data[et].test_mask.sum())
F_c   = int(data["compound"].x.shape[1])
F_t   = int(data["target"].x.shape[1])     # one-hot part only
F_e   = int(data[et].edge_attr.shape[1])

train_labels = data[et].edge_label[data[et].train_mask]
val_labels   = data[et].edge_label[data[et].val_mask]
test_labels  = data[et].edge_label[data[et].test_mask]
train_mean   = float(train_labels.mean())
train_std    = float(train_labels.std())
pchembl_min  = float(data[et].edge_label.min())
pchembl_max  = float(data[et].edge_label.max())
baseline_val  = float(torch.sqrt(((val_labels  - train_mean)**2).mean()))
baseline_test = float(torch.sqrt(((test_labels - train_mean)**2).mean()))

# Parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_params_enc_c = sum(p.numel() for p in model.compound_encoder.parameters())
n_params_enc_t = sum(p.numel() for p in model.target_encoder.parameters()) + \
                 sum(p.numel() for p in model.target_embedding.parameters())
n_params_conv  = sum(p.numel() for p in model.convs.parameters()) + \
                 sum(p.numel() for p in model.bns_compound.parameters()) + \
                 sum(p.numel() for p in model.bns_target.parameters())
n_params_head  = sum(p.numel() for p in model.edge_head.parameters())

# Training numbers
iters_per_epoch = math.ceil(E_tr / BATCH_SIZE)
max_iters       = MAX_EPOCHS * iters_per_epoch

# Year info
act_df = meta["activities_df"]
train_years = act_df.loc[data[et].train_mask.numpy().astype(bool), "doc_year"].dropna()
test_years  = act_df.loc[data[et].test_mask.numpy().astype(bool),  "doc_year"].dropna()
train_yr_min = int(train_years.min()) if len(train_years) else 1977
train_yr_max = int(train_years.max()) if len(train_years) else 2014
test_yr_min  = int(test_years.min())  if len(test_years)  else TEMPORAL_CUTOFF_YEAR
test_yr_max  = int(test_years.max())  if len(test_years)  else 2024

print("Generating HTML report …")

# ───────────────────────────────────────────────────────────────────────────
# 2.  Build HTML
# ───────────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>OpioidGNN — דוח מלא</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root {{
    --blue:   #1a73e8;
    --green:  #34a853;
    --orange: #ea8600;
    --red:    #d93025;
    --purple: #7b2d8b;
    --gray:   #5f6368;
    --bg:     #f8f9fa;
    --card:   #ffffff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; background: var(--bg);
          color: #202124; font-size: 15px; line-height: 1.6; }}
  h1 {{ font-size: 2rem; color: var(--blue); padding: 32px 40px 8px;
        border-bottom: 3px solid var(--blue); }}
  h2 {{ font-size: 1.35rem; color: var(--blue); margin: 28px 0 14px;
        padding-bottom: 6px; border-bottom: 1px solid #dadce0; }}
  h3 {{ font-size: 1.1rem; color: var(--gray); margin: 18px 0 8px; }}
  .subtitle {{ color: var(--gray); padding: 4px 40px 20px; font-size: 0.95rem; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 0 40px 60px; }}
  .card {{ background: var(--card); border-radius: 10px; padding: 24px 28px;
           margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(2,1fr); gap: 16px; }}
  .stat-box {{ background: var(--card); border-radius: 10px; padding: 20px;
               box-shadow: 0 1px 4px rgba(0,0,0,.12); text-align: center; }}
  .stat-val  {{ font-size: 2rem; font-weight: 700; }}
  .stat-lbl  {{ font-size: 0.82rem; color: var(--gray); margin-top: 4px; }}
  .c-blue   {{ color: var(--blue); }}
  .c-green  {{ color: var(--green); }}
  .c-orange {{ color: var(--orange); }}
  .c-red    {{ color: var(--red); }}
  .c-purple {{ color: var(--purple); }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }}
  th {{ background: #e8f0fe; color: var(--blue); text-align: right;
        padding: 9px 14px; font-weight: 600; border-bottom: 2px solid #c5d8fc; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #f1f3f4; vertical-align: top; }}
  tr:hover td {{ background: #f8f9fa; }}
  .math-box {{ background: #f8f9fa; border-right: 4px solid var(--blue);
               padding: 14px 18px; margin: 12px 0; border-radius: 4px;
               font-size: 0.95rem; }}
  .arch {{ display: flex; align-items: center; gap: 0; flex-wrap: wrap;
           justify-content: center; margin: 20px 0; }}
  .arch-block {{ border-radius: 8px; padding: 12px 18px; text-align: center;
                 font-size: 0.82rem; font-weight: 600; min-width: 110px; }}
  .arch-arrow {{ font-size: 1.4rem; color: var(--gray); padding: 0 4px; align-self: center; }}
  .ab-input  {{ background: #e8f0fe; border: 2px solid var(--blue); color: var(--blue); }}
  .ab-enc    {{ background: #e6f4ea; border: 2px solid var(--green); color: #1e7e34; }}
  .ab-gnn    {{ background: #fce8e6; border: 2px solid var(--red); color: var(--red); }}
  .ab-head   {{ background: #fef3e2; border: 2px solid var(--orange); color: #b06000; }}
  .ab-out    {{ background: #f3e8fd; border: 2px solid var(--purple); color: var(--purple); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.78rem; font-weight: 600; margin: 2px; }}
  .badge-blue   {{ background: #e8f0fe; color: var(--blue); }}
  .badge-green  {{ background: #e6f4ea; color: #1e7e34; }}
  .badge-orange {{ background: #fef3e2; color: #b06000; }}
  .badge-red    {{ background: #fce8e6; color: var(--red); }}
  .timeline {{ position: relative; padding-right: 28px; }}
  .tl-item {{ margin-bottom: 14px; padding-right: 20px; border-right: 3px solid #dadce0; }}
  .tl-item::before {{ content: "●"; position: absolute; right: -8px;
                      color: var(--blue); font-size: 1.1rem; }}
  .tl-title {{ font-weight: 600; color: #202124; }}
  .tl-desc  {{ font-size: 0.88rem; color: var(--gray); }}
  .split-bar {{ display: flex; border-radius: 8px; overflow: hidden;
                height: 38px; margin: 16px 0; font-size: 0.85rem;
                font-weight: 600; color: white; }}
  .sp-train {{ background: var(--blue);   display: flex; align-items: center;
               justify-content: center; }}
  .sp-val   {{ background: var(--orange); display: flex; align-items: center;
               justify-content: center; }}
  .sp-test  {{ background: var(--red);    display: flex; align-items: center;
               justify-content: center; }}
  code {{ background: #f1f3f4; padding: 1px 5px; border-radius: 4px;
          font-size: 0.88em; font-family: "Consolas", monospace; }}
  .note {{ background: #e8f0fe; border-radius: 6px; padding: 10px 16px;
           font-size: 0.88rem; color: #174ea6; margin: 10px 0; }}
  footer {{ text-align: center; color: var(--gray); font-size: 0.82rem;
            padding: 24px; border-top: 1px solid #dadce0; margin-top: 40px; }}
</style>
</head>
<body>
<h1>OpioidGNN — דוח מלא</h1>
<p class="subtitle">גרף ידע ODO · חיזוי ערכי pChEMBL בין תרכובות לקולטנים אופיואידיים</p>

<div class="container">

<!-- ═══════════════════════════  SECTION 1 — DATA  ══════════════════════ -->
<h2>1. נתונים</h2>

<div class="grid-3">
  <div class="stat-box">
    <div class="stat-val c-blue">{N_c:,}</div>
    <div class="stat-lbl">תרכובות כימיות<br>(Compound nodes)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val c-green">{N_t}</div>
    <div class="stat-lbl">מטרות אופיואידיות<br>(Target nodes)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val c-orange">{E:,}</div>
    <div class="stat-lbl">מדידות ביו-פעילות<br>(Activity edges)</div>
  </div>
</div>

<div class="card">
  <h3>פיצול זמני (Temporal Split) — גבול: {TEMPORAL_CUTOFF_YEAR}</h3>
  <p style="font-size:0.88rem;color:var(--gray);margin-bottom:12px;">
    מאמרים שפורסמו לפני {TEMPORAL_CUTOFF_YEAR} = אימון &amp; ולידציה ·
    מאמרים שפורסמו {TEMPORAL_CUTOFF_YEAR} ואילך = מבחן (העתיד הלא נראה)
  </p>

  <div class="split-bar">
    <div class="sp-train" style="width:{100*E_tr/E:.1f}%">
      Train {E_tr:,} ({100*E_tr/E:.1f}%)</div>
    <div class="sp-val" style="width:{100*E_val/E:.1f}%">
      Val {E_val:,} ({100*E_val/E:.1f}%)</div>
    <div class="sp-test" style="width:{100*E_te/E:.1f}%">
      Test {E_te:,} ({100*E_te/E:.1f}%)</div>
  </div>

  <table>
    <tr><th>Split</th><th>פעילויות</th><th>%</th><th>שנים</th><th>pChEMBL ממוצע</th><th>pChEMBL std</th></tr>
    <tr>
      <td><span class="badge badge-blue">Train</span></td>
      <td>{E_tr:,}</td><td>{100*E_tr/E:.1f}%</td>
      <td>{train_yr_min}–{train_yr_max}</td>
      <td>{float(train_labels.mean()):.3f}</td>
      <td>{float(train_labels.std()):.3f}</td>
    </tr>
    <tr>
      <td><span class="badge badge-orange">Val</span></td>
      <td>{E_val:,}</td><td>{100*E_val/E:.1f}%</td>
      <td>אקראי מה-Train</td>
      <td>{float(val_labels.mean()):.3f}</td>
      <td>{float(val_labels.std()):.3f}</td>
    </tr>
    <tr>
      <td><span class="badge badge-red">Test</span></td>
      <td>{E_te:,}</td><td>{100*E_te/E:.1f}%</td>
      <td>{test_yr_min}–{test_yr_max}</td>
      <td>{float(test_labels.mean()):.3f}</td>
      <td>{float(test_labels.std()):.3f}</td>
    </tr>
  </table>

  <div class="note">
    ⚠ פעילויות ללא שנת פרסום ({E - E_te - E_tr - E_val + E_tr + E_val - (E - E_te):,}
    חסרות שנה) הוקצו ל-Train.
    baseline RMSE (מנבא ממוצע): Val = <strong>{baseline_val:.4f}</strong> ·
    Test = <strong>{baseline_test:.4f}</strong> — המודל חייב לשפר על מספרים אלו.
  </div>
</div>

<div class="card">
  <h3>התפלגות pChEMBL (ציר ה-Y)</h3>
  <p style="font-size:0.88rem;margin-bottom:10px;">
    pChEMBL = −log₁₀(IC₅₀ / Ki / EC₅₀ [מולרי]). ערך גבוה = זיקה חזקה יותר.
    טווח בנתונים: <strong>{pchembl_min:.1f}–{pchembl_max:.1f}</strong> ·
    ממוצע train: <strong>{train_mean:.3f}</strong> ·
    סטיית תקן train: <strong>{train_std:.3f}</strong>
  </p>
  <div class="math-box">
    $$\\text{{pChEMBL}} = -\\log_{{10}}\\!\\left(\\frac{{\\text{{IC}}_{{50}} \\text{{ [or Ki/EC}}_{{50}}\\text{{]}}}}{{1}}\\ [\\text{{mol/L}}]\\right)$$
    <br>
    דוגמה: IC₅₀ = 1 nM = 10⁻⁹ M &nbsp;⟹&nbsp; pChEMBL = 9.0
  </div>
</div>

<!-- ═══════════════════════════  SECTION 2 — FEATURES  ══════════════════ -->
<h2>2. הנדסת Features</h2>

<div class="grid-2">
  <div class="card">
    <h3>תרכובות — {F_c:,} ממדים סה"כ</h3>
    <table>
      <tr><th>קבוצה</th><th>ממדים</th><th>תיאור</th></tr>
      <tr>
        <td><span class="badge badge-blue">Morgan ECFP4</span></td>
        <td><strong>{MORGAN_BITS:,}</strong></td>
        <td>Circular fingerprint, radius={MORGAN_RADIUS}, nBits={MORGAN_BITS} — מייצג את המבנה הכימי כווקטור בינארי</td>
      </tr>
      <tr>
        <td><span class="badge badge-green">ADMET (QikProp)</span></td>
        <td><strong>10</strong></td>
        <td>MW, aLogP, Ro5-violations, logPow, logS, H-donors, H-acceptors, logKhsa, oral-absorption%, SASA</td>
      </tr>
      <tr>
        <td><span class="badge badge-orange">קליני</span></td>
        <td><strong>2</strong></td>
        <td>maxPhase ÷ 4 (שלב קליני 0–4 מנורמל), isRadiolabeled (0/1)</td>
      </tr>
      <tr style="font-weight:600;">
        <td>סה"כ</td><td>{F_c:,}</td><td>לאחר StandardScaler על ADMET</td>
      </tr>
    </table>
    <div class="math-box" style="font-size:0.83rem;">
      Morgan fingerprint: כל bit מייצג תת-מבנה כימי ברדיוס r≤{MORGAN_RADIUS} סביב כל אטום<br>
      $$f_{{\\text{{Morgan}}}} \\in \\{{0,1\\}}^{{{MORGAN_BITS}}}$$
      ADMET נורמל: $$\\tilde{{x}} = \\frac{{x - \\mu_{{\\text{{train}}}}}}{{\\sigma_{{\\text{{train}}}}}}$$
    </div>
  </div>

  <div class="card">
    <h3>מטרות — {F_t} + 32 embedding ממדים</h3>
    <table>
      <tr><th>קבוצה</th><th>ממדים</th><th>ערכים</th></tr>
      <tr>
        <td><span class="badge badge-blue">targetType</span></td>
        <td><strong>4</strong></td>
        <td>Single protein, Protein family, Selectivity group, other</td>
      </tr>
      <tr>
        <td><span class="badge badge-green">targetSpecies</span></td>
        <td><strong>13</strong></td>
        <td>Homo sapiens, Rattus norvegicus, Mus musculus, … (12 מינים + other)</td>
      </tr>
      <tr>
        <td><span class="badge badge-purple">Learned Embedding</span></td>
        <td><strong>32</strong></td>
        <td>Embedding({N_t}, 32) — נלמד במהלך האימון עבור כל אחת מ-{N_t} המטרות</td>
      </tr>
      <tr style="font-weight:600;"><td>קלט ל-Encoder</td><td>49</td><td>17 one-hot + 32 embedding</td></tr>
    </table>

    <h3 style="margin-top:18px;">צלעות (Activity) — {F_e} ממדים</h3>
    <table>
      <tr><th>שדה</th><th>ממדים</th><th>ערכים</th></tr>
      <tr><td>endpointType</td><td>7</td><td>Ki, IC₅₀, EC₅₀, Inhibition, Activity, Binding, other</td></tr>
      <tr><td>qualifier</td><td>6</td><td>=, &lt;, &gt;, ≤, ≥, other</td></tr>
      <tr><td>experimentalSetting</td><td>3</td><td>in vitro, in vivo, other</td></tr>
      <tr style="font-weight:600;"><td>סה"כ</td><td>{F_e}</td><td>one-hot encoding</td></tr>
    </table>
  </div>
</div>

<!-- ═══════════════════════════  SECTION 3 — ARCHITECTURE  ══════════════ -->
<h2>3. ארכיטקטורת המודל — OpioidGNN</h2>

<div class="card">
  <h3>זרימת מידע — גרף דו-צדדי (Bipartite)</h3>
  <div class="arch">
    <div class="arch-block ab-input">Compound<br><small>x ∈ ℝ<sup>{F_c}</sup></small></div>
    <div class="arch-arrow">→</div>
    <div class="arch-block ab-enc">Compound<br>Encoder<br><small>→ ℝ<sup>256</sup></small></div>
    <div class="arch-arrow">↕</div>
    <div class="arch-block ab-gnn">Bipartite<br>GraphSAGE<br><small>×{GNN_LAYERS} layers</small></div>
    <div class="arch-arrow">→</div>
    <div class="arch-block ab-head">Edge<br>Head MLP</div>
    <div class="arch-arrow">→</div>
    <div class="arch-block ab-out">ŷ<br><small>pChEMBL ∈ ℝ</small></div>
  </div>
  <div class="arch" style="margin-top:0;">
    <div class="arch-block ab-input">Target<br><small>x∈ℝ<sup>17</sup>, idx∈ℤ</small></div>
    <div class="arch-arrow">→</div>
    <div class="arch-block ab-enc">Target<br>Encoder<br><small>→ ℝ<sup>256</sup></small></div>
    <div class="arch-arrow">↑</div>
    <div style="width:110px;"></div>
  </div>
</div>

<div class="card">
  <h3>פירוט שכבות ופרמטרים</h3>
  <table>
    <tr><th>שכבה</th><th>אופרציה</th><th>קלט → פלט</th><th>פרמטרים</th></tr>
    <tr>
      <td><strong>Compound Encoder</strong></td>
      <td>Linear → BatchNorm → ReLU → Dropout({DROPOUT})</td>
      <td>ℝ<sup>{F_c}</sup> → ℝ<sup>256</sup></td>
      <td>{n_params_enc_c:,}</td>
    </tr>
    <tr>
      <td><strong>Target Embedding</strong></td>
      <td>Embedding({N_t}, 32)</td>
      <td>ℤ<sup>1</sup> → ℝ<sup>32</sup></td>
      <td rowspan="2">{n_params_enc_t:,}</td>
    </tr>
    <tr>
      <td><strong>Target Encoder</strong></td>
      <td>Linear → BatchNorm → ReLU → Dropout({DROPOUT})</td>
      <td>ℝ<sup>49</sup> → ℝ<sup>256</sup></td>
    </tr>
    <tr>
      <td><strong>GNN Layer × {GNN_LAYERS}</strong></td>
      <td>HeteroConv(SAGEConv) + BN + ReLU + Dropout({DROPOUT})</td>
      <td>ℝ<sup>256</sup> ↔ ℝ<sup>256</sup></td>
      <td>{n_params_conv:,}</td>
    </tr>
    <tr>
      <td><strong>Edge Head MLP</strong></td>
      <td>Linear(528→256) → BN → ReLU → Linear(256→128) → BN → ReLU → Linear(128→1)</td>
      <td>ℝ<sup>{256+256+F_e}</sup> → ℝ<sup>1</sup></td>
      <td>{n_params_head:,}</td>
    </tr>
    <tr style="font-weight:700;background:#f8f9fa;">
      <td colspan="3">סה"כ פרמטרים הניתנים לאימון</td>
      <td>{n_params:,}</td>
    </tr>
  </table>
</div>

<div class="card">
  <h3>Message Passing — GraphSAGE על גרף Bipartite (שלב l → l+1)</h3>

  <div class="math-box">
    <strong>כיוון Compound → Target:</strong><br>
    $$h_t^{{(l+1)}} = \\sigma\\!\\left(\\mathbf{{W}}_1^{{(l)}} \\cdot
      \\text{{CONCAT}}\\!\\left(h_t^{{(l)}},\\;
        \\frac{{1}}{{|\\mathcal{{N}}(t)|}}\\sum_{{c\\,\\in\\,\\mathcal{{N}}(t)}} h_c^{{(l)}}
      \\right)\\right)$$
  </div>
  <div class="math-box">
    <strong>כיוון Target → Compound (reverse edges):</strong><br>
    $$h_c^{{(l+1)}} = \\sigma\\!\\left(\\mathbf{{W}}_2^{{(l)}} \\cdot
      \\text{{CONCAT}}\\!\\left(h_c^{{(l)}},\\;
        \\frac{{1}}{{|\\mathcal{{N}}(c)|}}\\sum_{{t\\,\\in\\,\\mathcal{{N}}(c)}} h_t^{{(l)}}
      \\right)\\right)$$
  </div>

  <p style="font-size:0.88rem;color:var(--gray);margin-top:8px;">
    σ = ReLU + BatchNorm + Dropout({DROPOUT}) ·
    המחזור מתבצע <strong>{GNN_LAYERS} פעמים</strong> ·
    לאחר {GNN_LAYERS} שכבות כל צומת "רואה" שכנים עד מרחק {GNN_LAYERS}
  </p>

  <div class="math-box" style="margin-top:16px;">
    <strong>חיזוי פלט לכל צלע (c, t):</strong><br>
    $$\\hat{{y}}_{{c,t}} = \\text{{MLP}}_{{\\text{{head}}}}\\!\\left(
      \\text{{CONCAT}}\\!\\left(h_c^{{(L)}},\\; h_t^{{(L)}},\\; e_{{c,t}}\\right)
    \\right)$$
    כאשר \\(e_{{c,t}} \\in \\mathbb{{R}}^{{{F_e}}}\\) הוא וקטור features של הצלע,
    \\(L = {GNN_LAYERS}\\) מספר שכבות GNN
  </div>
</div>

<!-- ═══════════════════════════  SECTION 4 — TRAINING  ══════════════════ -->
<h2>4. תהליך האימון</h2>

<div class="grid-2">
  <div class="card">
    <h3>פונקציית Loss ומטריקות</h3>
    <div class="math-box">
      <strong>MSE Loss (אימון):</strong><br>
      $$\\mathcal{{L}} = \\frac{{1}}{{|\\mathcal{{B}}|}}
        \\sum_{{(c,t)\\,\\in\\,\\mathcal{{B}}}}
        \\left(\\hat{{y}}_{{c,t}} - y_{{c,t}}\\right)^2$$
    </div>
    <div class="math-box">
      <strong>RMSE (הערכה):</strong>
      $$\\text{{RMSE}} = \\sqrt{{\\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}}(\\hat{{y}}_i - y_i)^2}}$$
    </div>
    <div class="math-box">
      <strong>MAE:</strong>
      $$\\text{{MAE}} = \\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}}|\\hat{{y}}_i - y_i|$$
    </div>
    <div class="math-box">
      <strong>Pearson r:</strong>
      $$r = \\frac{{\\sum(\\hat{{y}}_i - \\bar{{\\hat{{y}}}})(y_i - \\bar{{y}})}}
              {{\\sqrt{{\\sum(\\hat{{y}}_i-\\bar{{\\hat{{y}}}})^2}}\\cdot\\sqrt{{\\sum(y_i-\\bar{{y}})^2}}}}$$
    </div>
    <div class="math-box">
      <strong>R²:</strong>
      $$R^2 = 1 - \\frac{{\\sum_i(y_i - \\hat{{y}}_i)^2}}{{\\sum_i(y_i - \\bar{{y}})^2}}$$
    </div>
    <p style="font-size:0.85rem;color:var(--gray);margin-top:10px;">
      Baseline (מנבא ממוצע): Val RMSE = <strong>{baseline_val:.4f}</strong>,
      Test RMSE = <strong>{baseline_test:.4f}</strong>
    </p>
  </div>

  <div class="card">
    <h3>הגדרות אופטימיזציה</h3>
    <table>
      <tr><th>פרמטר</th><th>ערך</th><th>סיבה</th></tr>
      <tr><td>Optimizer</td><td><code>Adam</code></td><td>מסתגל לכל פרמטר בנפרד</td></tr>
      <tr><td>Learning Rate</td><td><code>{LR:.0e}</code></td><td>נקודת התחלה סטנדרטית ל-GNN</td></tr>
      <tr><td>Weight Decay (L2)</td><td><code>{WEIGHT_DECAY:.0e}</code></td><td>רגולריזציה קלה נגד overfitting</td></tr>
      <tr><td>LR Scheduler</td><td><code>ReduceLROnPlateau</code></td><td>מחצין LR פי 2 כשה-Val RMSE תקוע</td></tr>
      <tr><td>LR Patience</td><td>{LR_PATIENCE} epochs</td><td>המתנה לפני הקטנת LR</td></tr>
      <tr><td>Min LR</td><td><code>1e-6</code></td><td>רצפת ה-LR</td></tr>
      <tr><td>Early Stopping</td><td>{PATIENCE} epochs</td><td>עצירה אם אין שיפור ב-Val RMSE</td></tr>
      <tr><td>Max Epochs</td><td>{MAX_EPOCHS}</td><td>תקרה בטיחותית</td></tr>
      <tr><td>Gradient Clip</td><td>norm ≤ 5.0</td><td>מונע exploding gradients</td></tr>
      <tr><td>Batch Size</td><td>{BATCH_SIZE} צלעות</td><td>mini-batch על edges</td></tr>
      <tr><td>Dropout</td><td>{DROPOUT}</td><td>לכל שכבה ב-MLP וב-GNN</td></tr>
      <tr><td>Output Bias Init</td><td>{train_mean:.3f}</td><td>ממוצע pChEMBL ב-Train — מונע loss גבוה בepoches ראשונים</td></tr>
    </table>
  </div>
</div>

<div class="card">
  <h3>מספרי האיטרציות</h3>
  <div class="grid-3">
    <div class="stat-box">
      <div class="stat-val c-blue">{iters_per_epoch}</div>
      <div class="stat-lbl">איטרציות לכל epoch<br>⌈{E_tr:,} ÷ {BATCH_SIZE}⌉</div>
    </div>
    <div class="stat-box">
      <div class="stat-val c-green">{MAX_EPOCHS}</div>
      <div class="stat-lbl">epoch מקסימום<br>(עצירה מוקדמת אפשרית)</div>
    </div>
    <div class="stat-box">
      <div class="stat-val c-orange">{max_iters:,}</div>
      <div class="stat-lbl">איטרציות מקסימום סה"כ<br>{iters_per_epoch} × {MAX_EPOCHS}</div>
    </div>
  </div>
  <div class="math-box" style="margin-top:16px;">
    <strong>Adam Update Rule (לכל פרמטר θ):</strong>
    $$m_t = \\beta_1 m_{{t-1}} + (1-\\beta_1)\\nabla\\mathcal{{L}} \\qquad
      v_t = \\beta_2 v_{{t-1}} + (1-\\beta_2)(\\nabla\\mathcal{{L}})^2$$
    $$\\theta_{{t+1}} = \\theta_t - \\frac{{\\alpha}}{{\\sqrt{{\\hat{{v}}_t}} + \\epsilon}} \\hat{{m}}_t
    \\qquad (\\alpha={LR:.0e},\\; \\beta_1=0.9,\\; \\beta_2=0.999,\\; \\epsilon=10^{{-8}})$$
  </div>
</div>

<div class="card">
  <h3>זרימת האימון — Epoch by Epoch</h3>
  <div class="timeline">
    <div class="tl-item">
      <div class="tl-title">הכנה חד-פעמית לפני האימון</div>
      <div class="tl-desc">בניית גרף HeteroData · חישוב Morgan fingerprints · StandardScaler fit על Train · אתחול bias פלט = {train_mean:.3f}</div>
    </div>
    <div class="tl-item">
      <div class="tl-title">כל Epoch — שלב Train ({E_tr:,} צלעות)</div>
      <div class="tl-desc">
        1. ערבוב אקראי (shuffle) של {E_tr:,} צלעות<br>
        2. חלוקה ל-{iters_per_epoch} batches של {BATCH_SIZE} צלעות<br>
        3. לכל batch: Forward → MSE Loss → Backward → Adam step → Gradient clip (≤5.0)
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-title">כל Epoch — שלב Validation ({E_val:,} צלעות)</div>
      <div class="tl-desc">
        Forward pass ללא Dropout · חישוב RMSE, MAE, Pearson r, R² ·
        עדכון ReduceLROnPlateau לפי Val RMSE
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-title">שמירת Checkpoint</div>
      <div class="tl-desc">
        אם Val RMSE השתפר → שמור <code>gnn/checkpoints/best_model.pt</code> ·
        אחרת counter+1
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-title">Early Stopping</div>
      <div class="tl-desc">
        אם counter ≥ {PATIENCE} → טען best checkpoint ← הפסק אימון
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-title">הערכת Test ({E_te:,} צלעות · שנים {test_yr_min}–{test_yr_max})</div>
      <div class="tl-desc">
        Forward pass עם Best Model · פלט: RMSE, MAE, r, R² על "עתיד" שהמודל מעולם לא ראה
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════  SECTION 5 — WHY  ═══════════════════════ -->
<h2>5. מדוע הבחירות האלו?</h2>

<div class="grid-2">
  <div class="card">
    <h3>למה GraphSAGE ולא GCN?</h3>
    <p style="font-size:0.9rem;">
      GCN מניח גרף קבוע מראש — לא יכול להכליל לתרכובות חדשות (Inductive).
      GraphSAGE לומד <em>פונקציית אגרגציה</em> שמורחת לכל צומת, כולל חדשים.
      מכיוון שאנחנו מעריכים על תרכובות <em>ממאמרי 2015+</em> שיכולות להיות חדשות —
      Inductive learning הכרחי.
    </p>
  </div>
  <div class="card">
    <h3>למה Bipartite ולא גרף אחיד?</h3>
    <p style="font-size:0.9rem;">
      המשימה היא <em>חיזוי קשר תרכובת–מטרה</em> — גרף דו-צדדי טבעי לכך.
      Message passing בשני כיוונים מאפשר לתרכובות ל"ראות" מה נוסה על אותן מטרות,
      ולמטרות ל"ראות" אילו תרכובות קשורו אליהן — ייצוג משותף של מרחב הכימי והביולוגי.
    </p>
  </div>
  <div class="card">
    <h3>למה Temporal Split?</h3>
    <p style="font-size:0.9rem;">
      חלוקה אקראית גורמת ל-<em>data leakage</em>:
      מאמר מ-2018 בתרגול ממוצע את מאמר מ-2010 בולידציה.
      בפיצול זמני המודל מוערך על יכולתו לנבא מאמרים <em>עתידיים</em> —
      כפי שיידרש בפועל בגילוי תרופות.
    </p>
  </div>
  <div class="card">
    <h3>למה pChEMBL כ-target?</h3>
    <p style="font-size:0.9rem;">
      ערכי Ki, IC₅₀, EC₅₀ מגיעים בסקאלות שונות (nM, µM, %).
      pChEMBL מנרמל הכל לסקאלה אחידה (לוגריתמית, 3–12).
      זה מאפשר לאמן מודל אחד על <em>כל</em> סוגי המדידות ביחד.
    </p>
  </div>
</div>

<!-- ═══════════════════════════  SECTION 6 — SUMMARY  ══════════════════ -->
<h2>6. תמצית</h2>
<div class="card">
  <table>
    <tr><th>פרמטר</th><th>ערך</th></tr>
    <tr><td>משימה</td><td>רגרסיית pChEMBL (זיקה כימית)</td></tr>
    <tr><td>גרף</td><td>Bipartite Compound–Target, HeteroData (PyG)</td></tr>
    <tr><td>פיצול</td><td>Temporal: Train 1977–2014 / Test 2015–2020</td></tr>
    <tr><td>תרכובות / מטרות / פעילויות</td><td>{N_c:,} / {N_t} / {E:,}</td></tr>
    <tr><td>ממדי features</td><td>Compound {F_c:,} · Target 49 · Edge {F_e}</td></tr>
    <tr><td>ארכיטקטורה</td><td>Encoders → {GNN_LAYERS}×Bipartite-SAGEConv → Edge-MLP</td></tr>
    <tr><td>Hidden dim</td><td>256 בכל השכבות</td></tr>
    <tr><td>פרמטרים</td><td>{n_params:,}</td></tr>
    <tr><td>Loss</td><td>MSELoss(ŷ, pChEMBL)</td></tr>
    <tr><td>Optimizer</td><td>Adam (lr={LR:.0e}, wd={WEIGHT_DECAY:.0e})</td></tr>
    <tr><td>Epochs / Batch / Iters</td><td>max {MAX_EPOCHS} · {BATCH_SIZE} edges · {iters_per_epoch} iters/epoch</td></tr>
    <tr><td>Baseline RMSE לניצחון</td><td>Val {baseline_val:.4f} · Test {baseline_test:.4f}</td></tr>
    <tr><td>הרצה</td>
      <td><code>conda run -n odo python3 train_gnn.py</code></td></tr>
  </table>
</div>

</div><!-- /container -->

<footer>
  ODO Knowledge Graph · OpioidGNN · נוצר אוטומטית מ-gnn/generate_model_report.py
</footer>

</body>
</html>"""

# ───────────────────────────────────────────────────────────────────────────
# 3.  Write file
# ───────────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_report.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"Done → {out}")
