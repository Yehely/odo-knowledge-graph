#!/usr/bin/env python3
"""
ChEMBL Fetcher – Web UI
========================
Run:  python3 app.py
Open: http://localhost:5050
"""

import math
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify, send_file, Response
import json

# Import the fetcher logic
from chembl_fetcher import (
    build_rows_for_molecule,
    resolve_smiles_to_chembl,
    COLUMN_ORDER,
    apply_excel_formatting,
    NLP_PREDICTED_COLS,
)
import pandas as pd

app = Flask(__name__)
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job tracking
jobs = {}

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML = r"""
<!DOCTYPE html>
<html lang="he" dir="rtl" id="html-root">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChEMBL Data Fetcher</title>
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --primary: #6c63ff;
    --primary-hover: #5a52d5;
    --success: #00c48c;
    --error: #ff6b6b;
    --text: #e4e4e7;
    --text-dim: #9ca3af;
    --input-bg: #12141c;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    direction: rtl;
  }

  .container {
    max-width: 800px;
    margin: 0 auto;
    padding: 40px 20px;
  }

  header {
    text-align: center;
    margin-bottom: 40px;
  }

  header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }

  header p {
    color: var(--text-dim);
    font-size: 0.95rem;
  }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
  }

  label {
    display: block;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text);
  }

  .label-hint {
    font-weight: 400;
    color: var(--text-dim);
    font-size: 0.8rem;
  }

  textarea, input[type="text"] {
    width: 100%;
    background: var(--input-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.9rem;
    padding: 14px;
    outline: none;
    transition: border-color 0.2s;
    direction: ltr;
    text-align: left;
  }

  textarea:focus, input[type="text"]:focus {
    border-color: var(--primary);
  }

  textarea {
    min-height: 120px;
    resize: vertical;
  }

  .row {
    display: flex;
    gap: 16px;
    margin-top: 16px;
  }

  .row > div { flex: 1; }

  .input-mode-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
  }

  .tab-btn {
    padding: 8px 20px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.2s;
  }

  .tab-btn.active {
    background: var(--primary);
    color: #fff;
    border-color: var(--primary);
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 14px 24px;
    border-radius: 12px;
    border: none;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-primary {
    background: var(--primary);
    color: #fff;
  }

  .btn-primary:hover:not(:disabled) {
    background: var(--primary-hover);
    transform: translateY(-1px);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-download {
    background: var(--success);
    color: #fff;
    margin-top: 12px;
  }

  .btn-download:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
  }

  /* Progress */
  .progress-area {
    display: none;
    margin-top: 20px;
  }

  .progress-area.visible { display: block; }

  .progress-bar-container {
    background: var(--input-bg);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-top: 12px;
  }

  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--success));
    border-radius: 8px;
    width: 0%;
    transition: width 0.4s ease;
  }

  .log-area {
    margin-top: 16px;
    background: var(--input-bg);
    border-radius: 10px;
    padding: 14px;
    max-height: 250px;
    overflow-y: auto;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    direction: ltr;
    text-align: left;
    color: var(--text-dim);
  }

  .log-area .log-success { color: var(--success); }
  .log-area .log-error { color: var(--error); }
  .log-area .log-info { color: var(--primary); }

  .result-summary {
    display: flex;
    gap: 16px;
    margin-top: 16px;
  }

  .stat-box {
    flex: 1;
    background: var(--input-bg);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }

  .stat-box .num {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
  }

  .stat-box .lbl {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 4px;
  }

  /* Preview table */
  #preview-table th {
    background: var(--card);
    color: var(--primary);
    padding: 8px 10px;
    position: sticky;
    top: 0;
    white-space: nowrap;
    border-bottom: 2px solid var(--primary);
    font-weight: 600;
  }
  #preview-table th.qikprop-col {
    background: #3d3000;
    color: #ffd966;
    border-bottom: 2px solid #ffd966;
  }
  #preview-table td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    max-width: 250px;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text-dim);
  }
  #preview-table td.qikprop-col {
    background: rgba(255, 217, 102, 0.07);
    color: #ffd966;
  }
  #preview-table tr:hover td {
    background: rgba(108,99,255,0.07);
    color: var(--text);
  }
  #preview-area > div {
    max-height: 350px;
    overflow: auto;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Light theme overrides */
  html.light {
    --bg: #f5f7fa;
    --card: #ffffff;
    --border: #d1d5db;
    --primary: #5b54d8;
    --primary-hover: #4a43c0;
    --success: #0a9e7e;
    --error: #dc3545;
    --text: #1a202c;
    --text-dim: #4a5568;
    --input-bg: #edf2f7;
  }
  html.light #preview-table th { background: var(--card); }
  html.light #preview-table th.qikprop-col {
    background: #fff3cd;
    color: #856404;
    border-bottom: 2px solid #ffc107;
  }
  html.light #preview-table td.qikprop-col {
    background: rgba(255, 193, 7, 0.12);
    color: #664d03;
  }
  html.light .ml-note {
    background: rgba(0, 100, 200, 0.06);
    border-color: rgba(0, 100, 200, 0.2);
    color: #1a5fa8;
  }
  html.light .ml-note strong { color: #0d47a1; }

  /* Language visibility */
  html[lang="en"] .lang-he { display: none !important; }
  html[lang="he"] .lang-en { display: none !important; }

  /* Header toggle controls */
  .header-controls {
    position: absolute;
    top: 0;
    right: 0;
    display: flex;
    gap: 8px;
  }
  .toggle-btn {
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text);
    cursor: pointer;
    font-size: 0.82rem;
    font-weight: 600;
    transition: all 0.2s;
    user-select: none;
  }
  .toggle-btn:hover {
    border-color: var(--primary);
    color: var(--primary);
  }

  /* ML/estimated columns footnote */
  .ml-note {
    margin-top: 28px;
    padding: 14px 18px;
    background: rgba(100, 180, 255, 0.06);
    border: 1px solid rgba(100, 180, 255, 0.25);
    border-radius: 10px;
    font-size: 0.78rem;
    color: #7ab8e8;
    line-height: 1.7;
  }
  .ml-note strong { color: #4da6ff; }
</style>
</head>
<body>
<div class="container">

  <header style="position:relative">
    <div class="header-controls">
      <button class="toggle-btn" id="lang-btn" onclick="toggleLang()">EN</button>
      <button class="toggle-btn" id="theme-btn" onclick="toggleTheme()">☀️</button>
    </div>
    <h1>ChEMBL Data Fetcher</h1>
    <p>
      <span class="lang-he">משיכת נתוני מולקולות אוטומטית מ-ChEMBL</span>
      <span class="lang-en">Automatic molecular data retrieval from ChEMBL</span>
    </p>
  </header>

  <div class="card">
    <div class="input-mode-tabs">
      <button class="tab-btn active" onclick="setMode('ids')">ChEMBL IDs</button>
      <button class="tab-btn" onclick="setMode('smiles')">SMILES</button>
    </div>

    <div id="input-ids">
      <label>
        <span class="lang-he">הכנס ChEMBL IDs</span><span class="lang-en">Enter ChEMBL IDs</span>
        <span class="label-hint"><span class="lang-he">(אחד בכל שורה)</span><span class="lang-en">(one per line)</span></span>
      </label>
      <textarea id="ids-input" placeholder="CHEMBL25&#10;CHEMBL59&#10;CHEMBL3086308"></textarea>
    </div>

    <div id="input-smiles" style="display:none">
      <label>
        <span class="lang-he">הכנס SMILES</span><span class="lang-en">Enter SMILES</span>
        <span class="label-hint"><span class="lang-he">(אחד בכל שורה)</span><span class="lang-en">(one per line)</span></span>
      </label>
      <textarea id="smiles-input" placeholder="CC(=O)Oc1ccccc1C(=O)O&#10;CN1C=NC2=C1C(=O)N(C(=O)N2C)C"></textarea>
    </div>

    <div class="row">
      <div>
        <label><span class="lang-he">סינון לפי Target</span><span class="lang-en">Filter by Target</span> <span class="label-hint"><span class="lang-he">(אופציונלי)</span><span class="lang-en">(optional)</span></span></label>
        <input type="text" id="target-input" placeholder="CHEMBL233">
      </div>
      <div>
        <label><span class="lang-he">שם קובץ פלט</span><span class="lang-en">Output filename</span> <span class="label-hint"><span class="lang-he">(אופציונלי)</span><span class="lang-en">(optional)</span></span></label>
        <input type="text" id="output-name" placeholder="chembl_output.xlsx">
      </div>
    </div>

    <div style="margin-top: 20px">
      <button class="btn btn-primary" id="run-btn" onclick="startJob()">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/><path d="M9 12l2 2 4-4"/></svg>
        <span class="lang-he">התחל משיכת נתונים</span><span class="lang-en">Start fetching</span>
      </button>
    </div>

    <div class="progress-area" id="progress-area">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span id="progress-status" style="font-weight:600;"></span>
        <span id="progress-pct" style="font-size:0.85rem; color:var(--text-dim);"></span>
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar" id="progress-bar"></div>
      </div>
      <div class="log-area" id="log-area"></div>
      <div class="result-summary" id="result-summary" style="display:none">
        <div class="stat-box"><div class="num" id="stat-rows">0</div><div class="lbl"><span class="lang-he">שורות</span><span class="lang-en">Rows</span></div></div>
        <div class="stat-box"><div class="num" id="stat-molecules">0</div><div class="lbl"><span class="lang-he">מולקולות</span><span class="lang-en">Molecules</span></div></div>
        <div class="stat-box"><div class="num" id="stat-coverage">—</div><div class="lbl"><span class="lang-he">כיסוי עמודות</span><span class="lang-en">Coverage</span></div></div>
        <div class="stat-box"><div class="num" id="stat-certainty">—</div><div class="lbl"><span class="lang-he">ודאות חילוץ</span><span class="lang-en">Certainty</span></div></div>
        <div class="stat-box" style="border:1px solid rgba(100,180,255,0.4);background:rgba(100,180,255,0.07)"><div class="num" id="stat-ml" style="color:#4da6ff">0</div><div class="lbl"><span class="lang-he">עמודות ML ⓘ</span><span class="lang-en">ML columns ⓘ</span></div></div>
      </div>
      <div id="preview-area" style="display:none; margin-top:16px;">
        <div style="overflow-x:auto; border-radius:10px; border:1px solid var(--border);">
          <table id="preview-table" style="width:100%; border-collapse:collapse; font-size:0.75rem; font-family:'SF Mono','Fira Code',monospace; direction:ltr; text-align:left;">
          </table>
        </div>
      </div>
      <div id="download-area" style="display:none">
        <button class="btn btn-download" id="download-btn" onclick="downloadFile()">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          <span class="lang-he">הורד קובץ Excel</span><span class="lang-en">Download Excel file</span>
        </button>
      </div>
    </div>
  </div>

  <div class="ml-note">
    <span class="lang-he">
      <strong>⚠ עמודות מוערכות — לא נלקחו ישירות ממאמרים</strong><br>
      חלק מהעמודות בקובץ האקסל אינן מחולצות ממאמרים או APIs אלא מוערכות ע"י מודלים ממוחשבים:<br>
      • <strong>עמודות qikprop_* (צהוב)</strong> — Random Forest על 217 דסקריפטורים מולקולריים (RDKit), אומן על 12,905 מולקולות.
      דיוק R²: SASA 0.95 · FISA 0.94 · HBA/HBD 0.97 · QPlogPw 0.97 · QPlogPo/w 0.92 · QPlogS 0.80 · QPlogKhsa 0.94 · %OralAbs 0.94.
      <em>ערכי Dipole אינם מחושבים (דורשים מכניקה קוונטית).</em><br>
      • <strong>עמודות ML (כחול)</strong> — assay_kit: RF text classifier (~2,678 דוגמאות) ·
      bao_reference_compound &amp; dose_reference_compound: חילוץ regex ·
      ncit_route_of_administration_id &amp; mi_database_citation_id: מיפוי סטטי שם→ID.<br>
      <em>אחוז הודאות = שיעור העמודות שנחולצו ישירות (ללא ML) מתוך סך העמודות המאוכלסות.</em>
    </span>
    <span class="lang-en">
      <strong>⚠ Estimated columns — not taken directly from papers</strong><br>
      Some Excel columns are not extracted from papers/APIs but are estimated by computational models:<br>
      • <strong>qikprop_* columns (yellow)</strong> — Random Forest on 217 RDKit molecular descriptors, trained on 12,905 molecules.
      Accuracy R²: SASA 0.95 · FISA 0.94 · HBA/HBD 0.97 · QPlogPw 0.97 · QPlogPo/w 0.92 · QPlogS 0.80 · QPlogKhsa 0.94 · %OralAbs 0.94.
      <em>Dipole values are not computed (require quantum mechanics).</em><br>
      • <strong>ML columns (blue)</strong> — assay_kit: RF text classifier (~2,678 examples) ·
      bao_reference_compound &amp; dose_reference_compound: regex extraction ·
      ncit_route_of_administration_id &amp; mi_database_citation_id: static name→ID lookup.<br>
      <em>Certainty % = fraction of populated columns extracted directly (without ML).</em>
    </span>
  </div>

</div>

<script>
let currentMode = 'ids';
let currentJobId = null;
let pollTimer = null;
let currentLang = localStorage.getItem('lang') || 'he';
let currentTheme = localStorage.getItem('theme') || 'dark';

function applyLang(lang) {
  const html = document.documentElement;
  html.lang = lang;
  html.dir = lang === 'he' ? 'rtl' : 'ltr';
  document.getElementById('lang-btn').textContent = lang === 'he' ? 'EN' : 'עב';
  localStorage.setItem('lang', lang);
  currentLang = lang;
}
function toggleLang() { applyLang(currentLang === 'he' ? 'en' : 'he'); }

function applyTheme(theme) {
  if (theme === 'light') {
    document.documentElement.classList.add('light');
    document.getElementById('theme-btn').textContent = '🌙';
  } else {
    document.documentElement.classList.remove('light');
    document.getElementById('theme-btn').textContent = '☀️';
  }
  localStorage.setItem('theme', theme);
  currentTheme = theme;
}
function toggleTheme() { applyTheme(currentTheme === 'dark' ? 'light' : 'dark'); }

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('input-ids').style.display = mode === 'ids' ? '' : 'none';
  document.getElementById('input-smiles').style.display = mode === 'smiles' ? '' : 'none';
}

function addLog(msg, cls) {
  const area = document.getElementById('log-area');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = msg;
  area.appendChild(line);
  area.scrollTop = area.scrollHeight;
}

function startJob() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;

  const ids = currentMode === 'ids'
    ? document.getElementById('ids-input').value.trim()
    : '';
  const smiles = currentMode === 'smiles'
    ? document.getElementById('smiles-input').value.trim()
    : '';
  const target = document.getElementById('target-input').value.trim();
  const output = document.getElementById('output-name').value.trim();

  if (!ids && !smiles) {
    alert(currentLang === 'he' ? 'הכנס לפחות מולקולה אחת' : 'Enter at least one molecule');
    btn.disabled = false;
    return;
  }

  // Reset UI
  const area = document.getElementById('progress-area');
  area.classList.add('visible');
  document.getElementById('log-area').innerHTML = '';
  document.getElementById('result-summary').style.display = 'none';
  document.getElementById('download-area').style.display = 'none';
  document.getElementById('progress-bar').style.width = '0%';
  document.getElementById('progress-pct').textContent = '';
  document.getElementById('progress-status').textContent = currentLang === 'he' ? 'מתחיל...' : 'Starting...';

  fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ ids, smiles, target, output })
  })
  .then(r => r.json())
  .then(data => {
    currentJobId = data.job_id;
    addLog('Job started: ' + currentJobId, 'log-info');
    pollTimer = setInterval(pollStatus, 1000);
  });
}

function pollStatus() {
  if (!currentJobId) return;
  fetch('/api/status/' + currentJobId)
  .then(r => r.json())
  .then(data => {
    // Update progress
    const pct = data.total > 0 ? Math.round((data.processed / data.total) * 100) : 0;
    document.getElementById('progress-bar').style.width = pct + '%';
    document.getElementById('progress-pct').textContent = pct + '%';

    if (data.status === 'running') {
      document.getElementById('progress-status').textContent = currentLang === 'he'
        ? 'מעבד ' + data.processed + ' / ' + data.total + ' מולקולות'
        : 'Processing ' + data.processed + ' / ' + data.total + ' molecules';
    }

    // Show new logs
    const area = document.getElementById('log-area');
    const currentLines = area.children.length;
    for (let i = currentLines; i < data.logs.length; i++) {
      const msg = data.logs[i];
      let cls = '';
      if (msg.includes('->') || msg.includes('Saved')) cls = 'log-success';
      else if (msg.includes('not found') || msg.includes('error') || msg.includes('No data')) cls = 'log-error';
      else if (msg.includes('Fetching') || msg.includes('Resolving') || msg.includes('Processing')) cls = 'log-info';
      addLog(msg, cls);
    }

    if (data.status === 'done') {
      clearInterval(pollTimer);
      document.getElementById('progress-status').textContent = currentLang === 'he' ? 'הושלם!' : 'Done!';
      document.getElementById('progress-bar').style.width = '100%';
      document.getElementById('progress-pct').textContent = '100%';
      document.getElementById('run-btn').disabled = false;

      if (data.rows > 0) {
        document.getElementById('stat-rows').textContent = data.rows;
        document.getElementById('stat-molecules').textContent = data.molecules;
        document.getElementById('stat-coverage').textContent = data.coverage_pct != null ? data.coverage_pct + '%' : '—';
        document.getElementById('stat-certainty').textContent = data.certainty_pct != null ? data.certainty_pct + '%' : '—';
        document.getElementById('stat-ml').textContent = data.ml_columns || '0';
        document.getElementById('result-summary').style.display = 'flex';
        document.getElementById('download-area').style.display = 'block';
        // Load preview
        fetch('/api/preview/' + currentJobId)
          .then(r => r.json())
          .then(preview => {
            if (preview.headers) renderPreview(preview);
          });
      }
    }

    if (data.status === 'error') {
      clearInterval(pollTimer);
      document.getElementById('progress-status').textContent = currentLang === 'he' ? 'שגיאה' : 'Error';
      document.getElementById('run-btn').disabled = false;
    }
  });
}

function renderPreview(preview) {
  const table = document.getElementById('preview-table');
  // Only show populated columns
  const popIdx = preview.headers.map((h, i) => i).filter(i => preview.populated[i]);
  let html = '<thead><tr>';
  popIdx.forEach(i => {
    const isQik = preview.headers[i].startsWith('qikprop_');
    const cls = isQik ? ' class="qikprop-col"' : '';
    const qikTitle = currentLang === 'he' ? '⚠ ערך מוערך — חושב ע\'י RDKit' : '⚠ Estimated value — computed by RDKit';
    const title = isQik ? (' title="' + qikTitle + '"') : '';
    html += '<th' + cls + title + '>' + (isQik ? '⚠ ' : '') + preview.headers[i] + '</th>';
  });
  html += '</tr></thead><tbody>';
  preview.rows.forEach(row => {
    html += '<tr>';
    popIdx.forEach(i => {
      const isQik = preview.headers[i].startsWith('qikprop_');
      const cls = isQik ? ' class="qikprop-col"' : '';
      const v = row[i];
      html += '<td' + cls + ' title="' + (v !== null ? String(v) : '') + '">' + (v !== null ? String(v) : '') + '</td>';
    });
    html += '</tr>';
  });
  html += '</tbody>';
  table.innerHTML = html;
  document.getElementById('preview-area').style.display = 'block';
}

function downloadFile() {
  if (currentJobId) {
    window.location.href = '/api/download/' + currentJobId;
  }
}

// Apply saved preferences on load
applyLang(currentLang);
applyTheme(currentTheme);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def run_job(job_id, chembl_ids, smiles_list, target_filter, output_name):
    job = jobs[job_id]
    resolved_ids = list(chembl_ids)

    # Resolve SMILES
    for smi in smiles_list:
        job["logs"].append(f"Resolving SMILES: {smi}")
        cid = resolve_smiles_to_chembl(smi)
        if cid:
            job["logs"].append(f"  -> {cid}")
            resolved_ids.append(cid)
        else:
            job["logs"].append(f"  -> Not found in ChEMBL")
        time.sleep(0.25)

    # Deduplicate
    seen = set()
    unique_ids = []
    for cid in resolved_ids:
        if cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)

    job["total"] = len(unique_ids)

    if not unique_ids:
        job["logs"].append("No molecules to process!")
        job["status"] = "error"
        return

    job["logs"].append(f"Fetching data for {len(unique_ids)} molecule(s)...")

    all_rows = []
    for i, cid in enumerate(unique_ids, 1):
        job["logs"].append(f"[{i}/{len(unique_ids)}] Processing {cid}")
        try:
            rows = build_rows_for_molecule(cid, target_filter=target_filter or None)
            all_rows.extend(rows)
            job["logs"].append(f"    -> {len(rows)} rows added")
        except Exception as e:
            job["logs"].append(f"    -> error: {e}")
        job["processed"] = i

    if not all_rows:
        job["logs"].append("No data found!")
        job["status"] = "error"
        return

    df = pd.DataFrame(all_rows)

    # Ensure all 131 columns exist
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = None

    # Reorder to match original dataset exactly
    extra_cols = [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[COLUMN_ORDER + extra_cols]

    fname = output_name or "chembl_output.xlsx"
    if not fname.endswith(".xlsx"):
        fname += ".xlsx"
    out_path = OUTPUT_DIR / f"{job_id}_{fname}"
    df.to_excel(out_path, index=False, engine="openpyxl")
    apply_excel_formatting(df, out_path)

    populated = sum(1 for c in COLUMN_ORDER if df[c].notna().any())
    qikprop_cols = [c for c in COLUMN_ORDER if c.startswith("qikprop_")]
    ml_cols = qikprop_cols + [c for c in NLP_PREDICTED_COLS if c in df.columns]
    ml_columns = sum(1 for c in ml_cols if df[c].notna().any())
    job["output_file"] = str(out_path)
    job["output_name"] = fname
    job["rows"] = len(df)
    job["molecules"] = df["chembl_compound_id"].nunique()
    job["columns"] = len(COLUMN_ORDER)
    job["populated"] = populated
    coverage_pct = round(populated / len(COLUMN_ORDER) * 100, 1)
    certainty_pct = round((populated - ml_columns) / populated * 100, 1) if populated > 0 else 100.0
    job["ml_columns"] = ml_columns
    job["coverage_pct"] = coverage_pct
    job["certainty_pct"] = certainty_pct
    job["logs"].append(f"Saved {len(df)} rows to {fname}")
    job["logs"].append(f"Coverage: {coverage_pct}% | Certainty: {certainty_pct}% | ML-estimated: {ml_columns} cols")
    job["status"] = "done"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.json
    ids_text = data.get("ids", "").strip()
    smiles_text = data.get("smiles", "").strip()
    target = data.get("target", "").strip()
    output = data.get("output", "").strip()

    chembl_ids = [
        line.strip().upper() for line in ids_text.splitlines()
        if line.strip() and line.strip().upper().startswith("CHEMBL")
    ]
    smiles_list = [
        line.strip() for line in smiles_text.splitlines()
        if line.strip()
    ]

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "running",
        "total": 0,
        "processed": 0,
        "logs": [],
        "rows": 0,
        "molecules": 0,
        "columns": 0,
        "ml_columns": 0,
        "coverage_pct": None,
        "certainty_pct": None,
        "output_file": None,
        "output_name": None,
    }

    t = threading.Thread(
        target=run_job,
        args=(job_id, chembl_ids, smiles_list, target, output),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


def _sanitize(val):
    """Convert NaN/Inf to None for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


@app.route("/api/preview/<job_id>")
def api_preview(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("output_file"):
        return jsonify({"error": "File not found"}), 404
    try:
        df = pd.read_excel(job["output_file"], engine="openpyxl")
        headers = list(df.columns)
        populated = [bool(df[c].notna().any()) for c in headers]
        # Sanitize each value to avoid NaN in JSON
        preview_rows = []
        for _, row in df.head(10).iterrows():
            preview_rows.append([_sanitize(v) for v in row.tolist()])
        payload = json.dumps({
            "headers": headers,
            "populated": populated,
            "rows": preview_rows,
        }, ensure_ascii=False)
        return Response(payload, mimetype="application/json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<job_id>")
def api_download(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("output_file"):
        return jsonify({"error": "File not found"}), 404
    return send_file(
        job["output_file"],
        as_attachment=True,
        download_name=job["output_name"],
    )


if __name__ == "__main__":
    print("\n  ChEMBL Data Fetcher UI")
    print("  http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
