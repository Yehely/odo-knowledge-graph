"""
setup_graphdb.py – Create the ODO GraphDB repository and import all TTL files.
Run after build_kg.py has generated the output/*.ttl files.
"""

import os
import glob
import json
import time
import requests

BASE_URL    = "http://localhost:7200"
REPO_ID     = "odo-kg"
REPO_TITLE  = "ODO Knowledge Graph – Opioid Drug-Receptor Interactions"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
ONTOLOGY    = os.path.join(os.path.dirname(__file__), "odo_ontology.ttl")


def create_repository():
    """Create GraphDB repository using the REST API (Turtle config, GraphDB 11.x)."""

    config_ttl = f"""
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep:   <http://www.openrdf.org/config/repository#> .
@prefix sr:    <http://www.openrdf.org/config/repository/sail#> .
@prefix sail:  <http://www.openrdf.org/config/sail#> .
@prefix owlim: <http://www.ontotext.com/trree/owlim#> .

[] a rep:Repository ;
   rep:repositoryID "{REPO_ID}" ;
   rdfs:label "{REPO_TITLE}" ;
   rep:repositoryImpl [
     rep:repositoryType "openrdf:SailRepository" ;
     sr:sailImpl [
       sail:sailType "owlim:Sail" ;
       owlim:ruleset "rdfsplus-optimized" ;
       owlim:storage-folder "storage" ;
       owlim:base-URL "http://odo-project.org/data#" ;
       owlim:entity-index-size "10000000"
     ]
   ] .
"""

    print(f"Creating repository '{REPO_ID}'…")
    r = requests.post(
        f"{BASE_URL}/rest/repositories",
        files={"config": ("config.ttl", config_ttl.encode("utf-8"), "text/turtle")},
    )
    if r.status_code in (200, 201):
        print(f"  Repository '{REPO_ID}' created.")
    elif r.status_code == 409:
        print(f"  Repository '{REPO_ID}' already exists – skipping creation.")
    else:
        print(f"  WARNING: status {r.status_code} – {r.text[:400]}")


def clear_repository():
    """Delete all statements from the repository to ensure a clean reimport."""
    print(f"Clearing all statements from '{REPO_ID}'…", end=" ")
    r = requests.delete(
        f"{BASE_URL}/repositories/{REPO_ID}/statements",
        timeout=120,
    )
    if r.status_code in (200, 204):
        print("OK")
    else:
        print(f"WARNING: HTTP {r.status_code} – {r.text[:200]}")


def import_ttl(filepath, label=""):
    """Import a Turtle file into the repository via REST."""
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  Importing {label or os.path.basename(filepath)} ({size_mb:.1f} MB)…", end=" ")

    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/repositories/{REPO_ID}/statements",
            data=f,
            headers={"Content-Type": "text/turtle; charset=UTF-8"},
            timeout=300,
        )
    if r.status_code in (200, 204):
        print("OK")
    else:
        print(f"FAILED (HTTP {r.status_code}): {r.text[:200]}")


def count_triples():
    """Run a SPARQL COUNT(*) and return the number of triples."""
    sparql = "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"
    r = requests.post(
        f"{BASE_URL}/repositories/{REPO_ID}",
        data=sparql,
        headers={
            "Content-Type":  "application/sparql-query",
            "Accept":        "application/sparql-results+json",
        },
        timeout=60,
    )
    if r.status_code == 200:
        return int(r.json()["results"]["bindings"][0]["n"]["value"])
    return -1


def run_sparql(query, label=""):
    """Execute a SPARQL SELECT and print results."""
    print(f"\n{'─'*60}")
    if label:
        print(f"Query: {label}")
    r = requests.post(
        f"{BASE_URL}/repositories/{REPO_ID}",
        data=query,
        headers={
            "Content-Type": "application/sparql-query",
            "Accept":       "application/sparql-results+json",
        },
        timeout=120,
    )
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    vars_ = data["head"]["vars"]
    bindings = data["results"]["bindings"]
    print("  " + " | ".join(f"{v:30}" for v in vars_))
    print("  " + "-" * (32 * len(vars_)))
    for b in bindings[:15]:
        row = " | ".join(f"{b.get(v, {}).get('value','')[:30]:30}" for v in vars_)
        print("  " + row)
    if len(bindings) > 15:
        print(f"  … ({len(bindings)} rows total, showing first 15)")


def main():
    # 1. Create repository (skipped if already exists)
    create_repository()
    time.sleep(2)

    # 1b. Clear any stale triples from previous runs
    print("\nClearing stale data…")
    clear_repository()

    # 2. Import ontology schema
    print("\nImporting ontology…")
    import_ttl(ONTOLOGY, "odo_ontology.ttl")

    # 3. Import data files in a logical order
    IMPORT_ORDER = [
        "compounds.ttl",
        "proteins_targets.ttl",
        "model_systems.ttl",
        "signaling.ttl",
        "documents.ttl",
        "assays.ttl",
        "activities.ttl",
    ]

    print("\nImporting data files…")
    for fname in IMPORT_ORDER:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            import_ttl(path, fname)
        else:
            print(f"  SKIPPING {fname} (file not found)")

    # 4. Verify
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    time.sleep(3)
    n = count_triples()
    print(f"\nTotal triples in repository: {n:,}")

    # 5. Summary queries
    run_sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?class (COUNT(?instance) AS ?count) WHERE {
  ?instance a ?class .
  FILTER (STRSTARTS(STR(?class), "http://odo-project.org/ontology#"))
} GROUP BY ?class ORDER BY DESC(?count)
""", "Entity counts per class")

    run_sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?targetName (COUNT(DISTINCT ?compound) AS ?numCompounds) WHERE {
  ?activity odo:hasCompound ?compound ;
            odo:hasTarget ?target .
  ?target odo:targetName ?targetName .
} GROUP BY ?targetName ORDER BY DESC(?numCompounds) LIMIT 10
""", "Top 10 targets by compound count")

    run_sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?endpointType (COUNT(*) AS ?count) WHERE {
  ?activity a odo:Activity ;
            odo:endpointType ?endpointType .
} GROUP BY ?endpointType ORDER BY DESC(?count) LIMIT 10
""", "Top 10 endpoint types")

    run_sparql("""
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?year (COUNT(DISTINCT ?doc) AS ?numDocs) WHERE {
  ?doc a odo:Document ;
       odo:documentYear ?year .
} GROUP BY ?year ORDER BY ?year
""", "Documents by publication year")

    print(f"\nDone! Open http://localhost:7200 and explore repository '{REPO_ID}'.")


if __name__ == "__main__":
    main()
