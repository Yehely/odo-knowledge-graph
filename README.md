# ODO Knowledge Graph – גרף ידע לאינטראקציות תרופות אופיואידיות

פרויקט סיום לתואר ראשון בביואינפורמטיקה.  
גרף ידע המבוסס על מסד נתונים של **37,362 מדידות ביולוגיות** על **13,350 תרכובות כימיות** שנבדקו על קולטנים אופיאידיים (mu, kappa, delta, NOP).

---

## מה הגרף מכיל

| ישות | כמות |
|---|---|
| תרכובות כימיות | 13,350 |
| מדידות פעילות (Ki / IC50 / EC50 / ...) | 37,362 |
| מאמרים מדעיים | 1,045 |
| קולטנים ויעדים ביולוגיים | 55 |
| חלבונים (GPCRs) | 12 |
| קווי תאים | 65 |
| רקמות | 20 |
| מינים ביולוגיים | 10 |
| **סה"כ triples** | **~870,000** |

הגרף מקושר לבסיסי נתונים חיצוניים: **ChEMBL**, **PubChem**, **UniProt**, **Cellosaurus**, **BTO**.

---

## דרישות מקדימות

- Python 3.8+
- Conda (מומלץ) עם סביבת `odo`:
  ```bash
  conda create -n odo python=3.10
  conda activate odo
  pip install pandas openpyxl rdflib requests
  ```
- [GraphDB Free](https://www.ontotext.com/products/graphdb/download/) מותקן ורץ על `localhost:7200`
- קובץ הנתונים: `Final_updated_Dataset_v2025_11-12.xlsx` (לא מסופק בגיטהאב)

---

## הרצה מהירה

```bash
# שלב 1 – יצירת קבצי RDF מתוך ה-Excel
conda run -n odo python3 build_kg.py

# שלב 2 – יצירת repository ב-GraphDB וייבוא הנתונים
conda run -n odo python3 setup_graphdb.py

# שלב 3 – אימות הנתונים (אופציונלי)
conda run -n odo python3 validate_kg.py
```

לאחר ההרצה, גלוש ל: **http://localhost:7200**

---

## מבנה הקבצים

```
knowledge-graph/
├── build_kg.py           # ETL: Excel → RDF Turtle (~870K triples)
├── setup_graphdb.py      # יצירת repository וייבוא ל-GraphDB
├── validate_kg.py        # 14 שאילתות SPARQL לאימות הנתונים
├── odo_ontology.ttl      # הגדרות האונטולוגיה (מחלקות ופרופרטיות)
└── output/               # קבצי RDF שנוצרים (נוצרים על ידי build_kg.py)
    ├── compounds.ttl
    ├── assays.ttl
    ├── activities.ttl
    ├── model_systems.ttl
    ├── proteins_targets.ttl
    ├── documents.ttl
    └── signaling.ttl
```

---

## אונטולוגיה – מחלקות עיקריות

```
odo:Compound        – תרכובת כימית
odo:Target          – יעד ביולוגי (קולטן)
odo:Protein         – חלבון ספציפי
odo:Assay           – ניסוי ביולוגי
odo:Activity        – תוצאת מדידה (Ki/IC50/EC50...)
odo:CellLine        – קו תאים (CHO, HEK293...)
odo:Tissue          – רקמה (brain, vas deferens...)
odo:Organism        – מין (Homo sapiens, Rattus norvegicus...)
odo:Document        – מאמר מדעי
odo:SignalingPathway – נתיב סיגנל תאי
```

Namespace: `http://odo-project.org/ontology#`

---

## דוגמת שאילתת SPARQL

```sparql
# תרכובות עם Ki < 1 nM על קולטן mu
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?chemblId ?ki WHERE {
  ?act odo:endpointType "Ki" ;
       odo:endpointValue ?ki ;
       odo:hasCompound ?c ;
       odo:hasTarget [ odo:targetName "Mu opioid receptor" ] .
  ?c odo:chemblId ?chemblId .
  FILTER (?ki < 1.0)
} ORDER BY ?ki LIMIT 20
```

הרץ שאילתות ב: **http://localhost:7200/sparql**

---

## רישיון

לשימוש אקדמי בלבד. הנתונים מבוססים על מסד הנתונים ODO 2025.
