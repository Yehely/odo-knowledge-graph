"""
validate_kg.py – SPARQL validation queries for the ODO Knowledge Graph.
Run after setup_graphdb.py to verify data quality and connectivity.
"""

import requests
import json

BASE_URL = "http://localhost:7200"
REPO_ID  = "odo-kg"
ENDPOINT = f"{BASE_URL}/repositories/{REPO_ID}"

HEADERS = {
    "Content-Type": "application/sparql-query",
    "Accept":       "application/sparql-results+json",
}


def sparql(query, label=""):
    """Execute SPARQL query and pretty-print results."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print("="*65)
    r = requests.post(ENDPOINT, data=query, headers=HEADERS, timeout=120)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:300]}")
        return []
    data = r.json()
    vars_ = data["head"]["vars"]
    bindings = data["results"]["bindings"]
    col_w = max(25, 60 // max(len(vars_), 1))
    header = " | ".join(f"{v:<{col_w}}" for v in vars_)
    print("  " + header)
    print("  " + "-" * len(header))
    for b in bindings[:20]:
        row = " | ".join(f"{b.get(v,{}).get('value','')[:col_w]:<{col_w}}" for v in vars_)
        print("  " + row)
    if len(bindings) > 20:
        print(f"  … ({len(bindings)} total rows, showing first 20)")
    return bindings


def main():
    print("\nODO Knowledge Graph – Validation Report")
    print("Repository:", REPO_ID)
    print("Endpoint:  ", ENDPOINT)

    # ── 1. Total triples ──────────────────────────────────────────────────
    sparql(
        "SELECT (COUNT(*) AS ?total) WHERE { ?s ?p ?o }",
        "1. Total triples in the graph"
    )

    # ── 2. Entity counts per class ────────────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?class (COUNT(?i) AS ?count) WHERE {
  ?i a ?class .
  FILTER (STRSTARTS(STR(?class), "http://odo-project.org/ontology#"))
} GROUP BY ?class ORDER BY DESC(?count)
""", "2. Entity counts per class")

    # ── 3. Sample compounds ───────────────────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?chemblId ?mw ?alogP ?maxPhase WHERE {
  ?c a odo:Compound ;
     odo:chemblId ?chemblId .
  OPTIONAL { ?c odo:molecularWeight ?mw }
  OPTIONAL { ?c odo:alogP ?alogP }
  OPTIONAL { ?c odo:maxPhase ?maxPhase }
} ORDER BY ?chemblId LIMIT 15
""", "3. Sample compounds with physicochemical properties")

    # ── 4. Most-tested targets ────────────────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?targetName ?targetType (COUNT(DISTINCT ?compound) AS ?nCompounds)
       (COUNT(?activity) AS ?nActivities) WHERE {
  ?activity odo:hasCompound ?compound ;
            odo:hasTarget   ?target .
  ?target odo:targetName ?targetName .
  OPTIONAL { ?target odo:targetType ?targetType }
} GROUP BY ?targetName ?targetType ORDER BY DESC(?nActivities) LIMIT 15
""", "4. Top targets by number of tested compounds & activities")

    # ── 5. Endpoint distribution ──────────────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?endpoint (COUNT(*) AS ?count) WHERE {
  ?a a odo:Activity ;
     odo:endpointType ?endpoint .
} GROUP BY ?endpoint ORDER BY DESC(?count) LIMIT 15
""", "5. Activity endpoint type distribution")

    # ── 6. High-affinity Mu-receptor binders (Ki < 1 nM) ─────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?chemblId ?targetName ?ki ?role WHERE {
  ?act a odo:Activity ;
       odo:endpointType "Ki" ;
       odo:endpointValue ?ki ;
       odo:hasCompound ?c ;
       odo:hasTarget ?t .
  ?c odo:chemblId ?chemblId .
  ?t odo:targetName ?targetName .
  FILTER (?ki < 1.0)
  FILTER CONTAINS(LCASE(?targetName), "mu")
  OPTIONAL { ?act odo:pharmacologicalRoleLabel ?role }
} ORDER BY ?ki LIMIT 20
""", "6. High-affinity Mu receptor binders (Ki < 1 nM)")

    # ── 7. Agonists vs antagonists per target ─────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?targetName ?role (COUNT(DISTINCT ?compound) AS ?n) WHERE {
  ?act odo:hasCompound ?compound ;
       odo:hasTarget ?target ;
       odo:pharmacologicalRoleLabel ?role .
  ?target odo:targetName ?targetName .
  FILTER (?role IN ("agonist", "antagonist", "partial agonist",
                    "inverse agonist", "positive allosteric modulator"))
} GROUP BY ?targetName ?role ORDER BY ?targetName ?role
""", "7. Pharmacological roles per target")

    # ── 8. Cell lines used in assays ──────────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?cellLineName (COUNT(DISTINCT ?assay) AS ?nAssays)
       (COUNT(DISTINCT ?compound) AS ?nCompounds) WHERE {
  ?assay odo:hasModelSystem ?ms .
  ?ms odo:usesCellLine ?cl .
  ?cl odo:cellLineName ?cellLineName .
  ?act odo:hasAssay ?assay ;
       odo:hasCompound ?compound .
} GROUP BY ?cellLineName ORDER BY DESC(?nAssays) LIMIT 15
""", "8. Cell lines: assay and compound coverage")

    # ── 9. Species distribution ───────────────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?species (COUNT(DISTINCT ?activity) AS ?nActivities) WHERE {
  ?activity odo:hasTarget ?target .
  ?target odo:targetSpecies ?species .
} GROUP BY ?species ORDER BY DESC(?nActivities) LIMIT 10
""", "9. Activities per species")

    # ── 10. Assays with most measurements ─────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?assayName ?experimentalSetting (COUNT(*) AS ?nActivities) WHERE {
  ?assay a odo:Assay .
  OPTIONAL { ?assay odo:assayName ?assayName }
  OPTIONAL { ?assay odo:experimentalSetting ?experimentalSetting }
  ?act odo:hasAssay ?assay .
} GROUP BY ?assayName ?experimentalSetting ORDER BY DESC(?nActivities) LIMIT 10
""", "10. Assays with most activity measurements")

    # ── 11. Publications timeline ─────────────────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?year (COUNT(DISTINCT ?doc) AS ?nDocs) WHERE {
  ?doc a odo:Document ;
       odo:documentYear ?year .
} GROUP BY ?year ORDER BY ?year
""", "11. Publications per year")

    # ── 12. Proteins with their UniProt IDs ───────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?proteinName ?uniprotId ?gpcr_category WHERE {
  ?p a odo:Protein .
  OPTIONAL { ?p odo:proteinName ?proteinName }
  OPTIONAL { ?p odo:uniprotId   ?uniprotId }
  OPTIONAL { ?p odo:gpcrCategory ?gpcr_category }
} ORDER BY ?proteinName
""", "12. Proteins with UniProt IDs and GPCR category")

    # ── 13. Full path: Compound → Assay → Target → Protein ───────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?chemblId ?assayName ?targetName ?proteinName ?uniprotId ?ki WHERE {
  ?act odo:hasCompound ?c ;
       odo:hasAssay ?assay ;
       odo:hasTarget ?target ;
       odo:endpointType "Ki" ;
       odo:endpointValue ?ki .
  ?c      odo:chemblId    ?chemblId .
  ?assay  odo:assayName   ?assayName .
  ?target odo:targetName  ?targetName .
  ?target odo:encodedBy   ?protein .
  ?protein odo:proteinName ?proteinName .
  OPTIONAL { ?protein odo:uniprotId ?uniprotId }
  FILTER (?ki < 5.0)
} ORDER BY ?ki LIMIT 15
""", "13. Full path: Compound → Assay → Target → Protein (Ki < 5 nM)")

    # ── 14. Linked Data – external URIs ──────────────────────────────────
    sparql("""
SELECT ?subject ?externalURI WHERE {
  ?subject <http://www.w3.org/2002/07/owl#sameAs> ?externalURI .
  FILTER (!STRSTARTS(STR(?externalURI), "http://odo-project.org"))
} LIMIT 20
""", "14. Sample owl:sameAs links to external databases")

    # ── 15. Targets without any linked protein ────────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?targetName ?chemblTargetId WHERE {
  ?target a odo:Target .
  OPTIONAL { ?target odo:targetName ?targetName }
  OPTIONAL { ?target odo:chemblTargetId ?chemblTargetId }
  FILTER NOT EXISTS { ?target odo:encodedBy ?protein }
} ORDER BY ?targetName
""", "15. Targets with no linked protein (possible source-data gaps)")

    # ── 16. Activities not linked to any compound ─────────────────────────
    sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT (COUNT(*) AS ?unlinked) WHERE {
  ?a a odo:Activity .
  FILTER NOT EXISTS { ?a odo:hasCompound ?c }
}
""", "16. Activity nodes missing odo:hasCompound (should be 0 after InChIKey fallback)")

    print("\n\nValidation complete.")
    print(f"Open http://localhost:7200/sparql for interactive queries.")


if __name__ == "__main__":
    main()
