# Knowledge Graph — `kg/`

System 1 of ODO's three systems (see the [Wiki](https://github.com/Yehely/odo-project/wiki) for the full narrative). Turns the ODO Excel dataset into a queryable RDF knowledge graph, in three stages, then serves as the substrate the [`gnn/`](../gnn/README.md)/[`hetero_gnn/`](../hetero_gnn/README.md) models train on.

## Modules

| File | Purpose |
|---|---|
| `build_kg.py` | ETL: Excel → 7 thematic RDF Turtle files in `output/` |
| `setup_graphdb.py` | Creates the `odo-kg` GraphDB repository and imports the Turtle files |
| `validate_kg.py` | 16 SPARQL validation checks against the loaded repository |
| `odo_ontology.ttl` | The OWL ontology — all classes and properties `build_kg.py` instantiates |

## Usage

```bash
conda run -n odo python3 kg/build_kg.py
conda run -n odo python3 kg/setup_graphdb.py   # requires GraphDB running on localhost:7200
conda run -n odo python3 kg/validate_kg.py
```

Full walkthrough (namespaces, example SPARQL queries, GraphDB install): [`docs/SETUP_GUIDE.md`](../docs/SETUP_GUIDE.md) and the developer-facing detail in [`CLAUDE.md`](../CLAUDE.md).

## Output

`output/` (the generated Turtle files) is gitignored — regenerate with `build_kg.py`.
