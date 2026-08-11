"""
generate_bipartite_viz.py
─────────────────────────
Generates bipartite_graph_interactive.html — an interactive vis.js visualization
of the ODO bipartite graph (Compound ↔ Target).

Filter: only exact measurements with pChEMBL ≥ 9 (high-affinity binders).

Each node and edge is hoverable and shows all its fields.
Clicking a node highlights its connections.

Run:
    conda run -n odo python3 generate_bipartite_viz.py
"""

import json
import re

import pandas as pd

OUTPUT = "bipartite_graph_interactive.html"
EXCEL  = "Final ODO Dataset_v2026-06-10.xlsx"
MIN_PCHEMBL = 9.0

COLS = [
    "chembl_compound_id", "rdkit_library_standard_inchi_key",
    "chembl_chemical_entity_name", "rdkit_canonical_smiles",
    "chembl_molecule_max_phase", "reference_radiolabeled_molecular_entity",
    "rdkit_molecular_weight", "chembl_alogp", "chembl_#ro5_violations",
    "qikprop_qplog_po/w", "qikprop_qplogs", "qikprop_donor_hb",
    "qikprop_accpt_hb", "qikprop_qplog_khsa",
    "qikprop_percent_human_oral_absorption", "qikprop_sasa",
    "chembl_target_id", "target_name", "target_type", "ncbi_target_taxonomy",
    "endpoint_qualifier", "pchembl_value", "endpoint",
    "bao_experimental_setting", "document_year",
    "compound_pharmacological_role",
]

TARGET_TYPE_COLOR = {
    "Single protein":    "#e74c3c",
    "Protein family":    "#2ecc71",
    "Selectivity group": "#3498db",
    "heteromer":         "#f39c12",
}


def safe(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def slugify(t):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(t).strip())


def flt(row, col):
    v = safe(row.get(col))
    try:
        return round(float(v), 2) if v else "—"
    except (ValueError, TypeError):
        return "—"


def tooltip_table(fields: dict) -> str:
    rows = "".join(
        f"<tr><td style='padding:2px 8px 2px 0;color:#8b949e'><b>{k}</b></td>"
        f"<td style='padding:2px 0'>{v}</td></tr>"
        for k, v in fields.items()
        if not k.startswith("_")
    )
    return (
        "<div style='font-family:monospace;font-size:11px;"
        "background:#1c2128;border-radius:4px;padding:6px'>"
        f"<table style='border-collapse:collapse'>{rows}</table></div>"
    )


def pchembl_color(pch: float) -> str:
    t = max(0.0, min(1.0, (pch - MIN_PCHEMBL) / 2.5))
    r = int(255 * (1 - t))
    g = int(130 + 125 * t)
    b = int(40 * (1 - t))
    return f"rgb({r},{g},{b})"


def build_graph(df: pd.DataFrame):
    # ── Compound table ─────────────────────────────────────────────────────
    seen_c: dict[str, int] = {}
    comp_data: dict[str, dict] = {}

    for _, row in df.iterrows():
        cid  = safe(row.get("chembl_compound_id"))
        ikey = safe(row.get("rdkit_library_standard_inchi_key"))
        key  = cid or (f"inchikey_{ikey}" if ikey else None)
        if not key or key in seen_c:
            continue
        seen_c[key] = len(seen_c)
        smiles = safe(row.get("rdkit_canonical_smiles")) or ""
        rl = safe(row.get("reference_radiolabeled_molecular_entity"))
        comp_data[key] = {
            "ChEMBL ID":           cid or "—",
            "Name":                safe(row.get("chembl_chemical_entity_name")) or cid or "—",
            "SMILES":              smiles[:55] + ("…" if len(smiles) > 55 else ""),
            "Max Phase":           flt(row, "chembl_molecule_max_phase"),
            "Radiolabeled":        "Yes" if rl and rl.lower() in ("true", "1", "yes", "t") else "No",
            "── ADMET ──":         "─────────────",
            "MW (Da)":             flt(row, "rdkit_molecular_weight"),
            "AlogP":               flt(row, "chembl_alogp"),
            "Ro5 Violations":      flt(row, "chembl_#ro5_violations"),
            "logPow (QP)":         flt(row, "qikprop_qplog_po/w"),
            "logS (QP)":           flt(row, "qikprop_qplogs"),
            "Donor HB":            flt(row, "qikprop_donor_hb"),
            "Acceptor HB":         flt(row, "qikprop_accpt_hb"),
            "logKhsa (QP)":        flt(row, "qikprop_qplog_khsa"),
            "Oral Absorption %":   flt(row, "qikprop_percent_human_oral_absorption"),
            "SASA (Å²)":           flt(row, "qikprop_sasa"),
        }

    # ── Target table ───────────────────────────────────────────────────────
    seen_t: dict[str, int] = {}
    targ_data: dict[str, dict] = {}

    for _, row in df.iterrows():
        tcid  = safe(row.get("chembl_target_id"))
        tname = safe(row.get("target_name"))
        key   = tcid or (slugify(tname) if tname else None)
        if not key or key in seen_t:
            continue
        seen_t[key] = len(seen_t)
        taxon = safe(row.get("ncbi_target_taxonomy")) or "Unknown"
        parts = taxon.strip().split()
        species = (
            parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])
            if len(parts) >= 2 else taxon.strip()
        )
        ttype = safe(row.get("target_type")) or "other"
        targ_data[key] = {
            "ChEMBL ID":   tcid or "—",
            "Name":        tname or "—",
            "Target Type": ttype,
            "Species":     species,
            "_color":      TARGET_TYPE_COLOR.get(ttype, "#95a5a6"),
        }

    # ── Edges ──────────────────────────────────────────────────────────────
    edges = []
    for _, row in df.iterrows():
        cid   = safe(row.get("chembl_compound_id"))
        ikey  = safe(row.get("rdkit_library_standard_inchi_key"))
        c_key = cid or (f"inchikey_{ikey}" if ikey else None)
        tcid  = safe(row.get("chembl_target_id"))
        tname = safe(row.get("target_name"))
        t_key = tcid or (slugify(tname) if tname else None)

        if not c_key or not t_key:
            continue
        if c_key not in seen_c or t_key not in seen_t:
            continue

        qual    = safe(row.get("endpoint_qualifier")) or ""
        pch_raw = safe(row.get("pchembl_value"))
        try:
            pch = float(pch_raw) if pch_raw else None
        except (ValueError, TypeError):
            pch = None

        if qual != "=" or pch is None or pch < MIN_PCHEMBL:
            continue

        year = safe(row.get("document_year"))
        try:
            year = int(float(year)) if year else "—"
        except (ValueError, TypeError):
            year = "—"

        edges.append({
            "from":             c_key,
            "to":               t_key,
            "pChEMBL Value":    round(pch, 2),
            "Qualifier":        qual,
            "Endpoint Type":    safe(row.get("endpoint")) or "—",
            "Exp. Setting":     safe(row.get("bao_experimental_setting")) or "—",
            "Pharmacol. Role":  safe(row.get("compound_pharmacological_role")) or "—",
            "Pub. Year":        year,
        })

    # Keep only compounds that appear in edges
    active_c = {e["from"] for e in edges}
    comp_data = {k: v for k, v in comp_data.items() if k in active_c}
    return comp_data, targ_data, edges, seen_t


def build_vis_objects(comp_data, targ_data, edges, seen_t):
    comp_keys = list(comp_data.keys())
    targ_keys  = list(targ_data.keys())
    N_COMP = len(comp_keys)
    N_TARG = len(targ_keys)

    # Edge counts per node (for sizing)
    comp_edge_count = {k: 0 for k in comp_keys}
    targ_edge_count = {k: 0 for k in targ_keys}
    for e in edges:
        if e["from"] in comp_edge_count:
            comp_edge_count[e["from"]] += 1
        if e["to"] in targ_edge_count:
            targ_edge_count[e["to"]] += 1

    vis_nodes = []

    # Compound nodes — left column
    for i, key in enumerate(comp_keys):
        d = comp_data[key]
        y = int((i / max(N_COMP - 1, 1)) * 3000 - 1500)
        size = min(8 + comp_edge_count[key] * 2, 28)
        vis_nodes.append({
            "id":    f"c_{key}",
            "label": d["Name"][:22],
            "title": tooltip_table(d),
            "group": "compound",
            "x": -700, "y": y,
            "size": size,
            "fixed": {"x": True},
            "color": {
                "background": "#2e86c1",
                "border":     "#1a5276",
                "highlight":  {"background": "#85c1e9", "border": "#1a5276"},
                "hover":      {"background": "#5dade2", "border": "#1a5276"},
            },
            "font": {"size": 9, "color": "#ecf0f1"},
            "shape": "dot",
        })

    # Target nodes — right column
    for i, key in enumerate(targ_keys):
        d = targ_data[key]
        y = int((i / max(N_TARG - 1, 1)) * 3000 - 1500)
        size = 16 + targ_edge_count.get(key, 0) // 8
        vis_nodes.append({
            "id":    f"t_{key}",
            "label": d["Name"][:28],
            "title": tooltip_table({k: v for k, v in d.items() if not k.startswith("_")}),
            "group": "target",
            "x": 700, "y": y,
            "size": size,
            "fixed": {"x": True},
            "color": {
                "background": d["_color"],
                "border":     "#111",
                "highlight":  {"background": "#f8c471", "border": "#111"},
                "hover":      {"background": "#f8c471", "border": "#111"},
            },
            "font": {"size": 11, "color": "#fff", "bold": True},
            "shape": "box",
        })

    vis_edges = []
    for e in edges:
        pch = e["pChEMBL Value"]
        col = pchembl_color(pch)
        edge_fields = {k: v for k, v in e.items() if k not in ("from", "to")}
        vis_edges.append({
            "from":  f"c_{e['from']}",
            "to":    f"t_{e['to']}",
            "title": tooltip_table(edge_fields),
            "color": {"color": col, "opacity": 0.55, "highlight": col, "hover": col},
            "width": round(0.6 + (pch - MIN_PCHEMBL) * 0.4, 2),
        })

    return vis_nodes, vis_edges


def render_html(vis_nodes, vis_edges, n_comp, n_targ, n_edges) -> str:
    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ODO Bipartite Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; font-family: 'Segoe UI', Arial, sans-serif; color: #e6edf3; }}

  #header {{
    padding: 10px 20px; background: #161b22;
    border-bottom: 1px solid #30363d;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }}
  #header h2 {{ font-size: 15px; font-weight: 600; }}
  .badge {{
    padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;
  }}
  .b-comp {{ background: #1a5276; color: #85c1e9; }}
  .b-targ {{ background: #641e16; color: #f1948a; }}
  .b-edge {{ background: #1e6b2e; color: #82e0aa; }}

  #legend {{
    padding: 6px 20px; background: #161b22;
    border-bottom: 1px solid #30363d;
    font-size: 11px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap;
  }}
  .dot {{
    width: 11px; height: 11px; border-radius: 50%;
    display: inline-block; margin-right: 4px; vertical-align: middle;
  }}
  .box-icon {{
    width: 14px; height: 10px; border-radius: 2px;
    display: inline-block; margin-right: 4px; vertical-align: middle;
  }}

  #network {{ width: 100%; height: calc(100vh - 86px); }}

  #panel {{
    position: fixed; top: 95px; right: 14px;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 10px 14px;
    font-size: 11px; color: #8b949e; width: 210px;
    line-height: 1.7;
  }}
  #panel b {{ color: #e6edf3; }}

  .vis-tooltip {{
    background: #1c2128 !important;
    border: 1px solid #444 !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    max-width: 380px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
  }}
</style>
</head>
<body>

<div id="header">
  <h2>🧬 ODO Bipartite Graph — Compound ↔ Target</h2>
  <span class="badge b-comp">⬤ {n_comp:,} Compounds</span>
  <span class="badge b-targ">■ {n_targ} Targets</span>
  <span class="badge b-edge">— {n_edges:,} Edges  (pChEMBL ≥ {MIN_PCHEMBL})</span>
</div>

<div id="legend">
  <span style="color:#8b949e">Target type:</span>
  <span><span class="box-icon" style="background:#e74c3c"></span>Single protein</span>
  <span><span class="box-icon" style="background:#2ecc71"></span>Protein family</span>
  <span><span class="box-icon" style="background:#3498db"></span>Selectivity group</span>
  <span><span class="box-icon" style="background:#f39c12"></span>Heteromer</span>
  <span><span class="box-icon" style="background:#95a5a6"></span>Other</span>
  &nbsp;|&nbsp;
  <span>Edge:
    <span style="color:#ff6b6b">■</span> pChEMBL 9
    →
    <span style="color:#51cf66">■</span> pChEMBL 11+
    (width ∝ affinity)
  </span>
  &nbsp;|&nbsp;
  <span style="color:#5dade2">⬤ Node size ∝ # experiments</span>
</div>

<div id="network"></div>

<div id="panel">
  <b>Navigation</b><br>
  🖱 Scroll → zoom<br>
  🖱 Drag → pan<br>
  👆 Click node → highlight its edges<br>
  👆 Click background → reset<br>
  🔍 Hover node/edge → full field details<br><br>
  <b>Layout</b><br>
  Left column: Compounds<br>
  Right column: Targets
</div>

<script>
var nodesData = {nodes_json};
var edgesData = {edges_json};

var nodes = new vis.DataSet(nodesData);
var edges = new vis.DataSet(edgesData);

var container = document.getElementById('network');
var options = {{
  physics: {{ enabled: false }},
  interaction: {{
    hover: true,
    tooltipDelay: 80,
    hideEdgesOnDrag: true,
    navigationButtons: false,
    keyboard: true,
  }},
  nodes: {{ borderWidth: 1, shadow: false }},
  edges: {{
    smooth: {{ type: 'curvedCW', roundness: 0.08 }},
    hoverWidth: 2.5,
    selectionWidth: 3,
    arrows: {{ to: {{ enabled: false }} }},
  }},
}};

var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);

// Click highlight
var originalNodeColors = {{}};
var originalEdgeColors = {{}};
nodesData.forEach(n => {{ originalNodeColors[n.id] = n.color; }});
edgesData.forEach(e => {{ originalEdgeColors[e.id || e.from+e.to] = e.color; }});

var highlighted = false;
network.on('click', function(params) {{
  if (params.nodes.length > 0) {{
    highlighted = true;
    var nodeId = params.nodes[0];
    var connectedEdges = network.getConnectedEdges(nodeId);
    var connectedNodes = network.getConnectedNodes(nodeId);
    connectedNodes.push(nodeId);

    nodes.update(nodes.getIds().map(id => ({{
      id,
      opacity: connectedNodes.includes(id) ? 1.0 : 0.05,
    }})));
    edges.update(edges.getIds().map(id => ({{
      id,
      color: {{ opacity: connectedEdges.includes(id) ? 0.9 : 0.02 }},
    }})));
  }} else if (params.edges.length === 0 && highlighted) {{
    highlighted = false;
    nodes.update(nodes.getIds().map(id => ({{ id, opacity: 1.0 }})));
    edges.update(edges.getIds().map(id => ({{ id, color: {{ opacity: 0.55 }} }})));
  }}
}});
</script>
</body>
</html>"""


def main():
    print("Reading Excel …")
    df = pd.read_excel(EXCEL, sheet_name="Full Dataset", usecols=COLS)
    print(f"  {len(df):,} rows loaded")

    print("Building graph …")
    comp_data, targ_data, edges, seen_t = build_graph(df)
    print(f"  Compounds : {len(comp_data):,}")
    print(f"  Targets   : {len(targ_data)}")
    print(f"  Edges     : {len(edges):,}  (pChEMBL ≥ {MIN_PCHEMBL})")

    print("Building vis.js objects …")
    vis_nodes, vis_edges = build_vis_objects(comp_data, targ_data, edges, seen_t)

    print("Writing HTML …")
    html = render_html(vis_nodes, vis_edges,
                       len(comp_data), len(targ_data), len(edges))
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone → {OUTPUT}")
    print("Open in browser:  xdg-open bipartite_graph_interactive.html")


if __name__ == "__main__":
    main()
