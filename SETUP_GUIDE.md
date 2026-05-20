# מדריך התקנה ותפעול – ODO Knowledge Graph

---

## שלב 1 – התקנת GraphDB

1. גש ל: https://www.ontotext.com/products/graphdb/download/
2. הורד **GraphDB Free** (גרסה 11.x)
3. התקן ולחץ **Start** – GraphDB יעלה בפורט **7200**
4. פתח דפדפן וגש ל: `http://localhost:7200`

---

## שלב 2 – הכנת סביבת Python

```bash
# יצירת סביבת conda
conda create -n odo python=3.10
conda activate odo

# התקנת חבילות
pip install pandas openpyxl rdflib requests
```

---

## שלב 3 – הורדת קבצי הפרויקט

```bash
git clone https://github.com/<username>/odo-knowledge-graph.git
cd odo-knowledge-graph
```

> הצב את קובץ ה-Excel `Final_updated_Dataset_v2025_11-12.xlsx` **בתוך תיקיית הפרויקט** (לא מסופק ב-GitHub).

---

## שלב 4 – בניית גרף הידע

```bash
conda run -n odo python3 build_kg.py
```

**מה קורה:** הסקריפט קורא את ה-Excel ומייצר 7 קבצי Turtle (RDF) בתיקיית `output/`.  
**זמן הרצה משוער:** 2–4 דקות  
**פלט צפוי:**
```
Processing rows: 100%|████████| 37362/37362
Compounds   : 12,906 nodes
Activities  : 37,346 nodes
Documents   :  1,165 nodes
...
Saved output/compounds.ttl (355,479 triples)
Saved output/activities.ttl (436,226 triples)
...
Total triples written: 869,134
```

---

## שלב 5 – ייבוא לגרפDB

ודא שGraphDB פועל ב-`http://localhost:7200`, ואז:

```bash
conda run -n odo python3 setup_graphdb.py
```

**מה קורה:** הסקריפט יוצר Repository בשם `odo-kg` ומייבא את כל הקבצים.  
**זמן הרצה משוער:** 3–5 דקות  
**פלט צפוי:**
```
Creating repository 'odo-kg'…
  Repository 'odo-kg' created.

Importing ontology…
  Importing odo_ontology.ttl (0.0 MB)… OK

Importing data files…
  Importing compounds.ttl (31.3 MB)… OK
  Importing activities.ttl (38.7 MB)… OK
  ...

Total triples in repository: 874,147
```

---

## שלב 6 – גלישה בגרף

### ממשק הרשת
פתח: **http://localhost:7200**

### ויזואליזציה
1. לחץ **Explore → Visual Graph**
2. בתיבת החיפוש הכתוב URI מלא, למשל:
   ```
   http://odo-project.org/data#compound_CHEMBL70
   ```
3. לחץ **Enter** – תראה את הצמתים המחוברים לתרכובת

### שאילתות SPARQL
1. לחץ **SPARQL** בתפריט השמאלי
2. בחר Repository: `odo-kg`
3. כתוב שאילתה ולחץ **Run**

**דוגמה – כמה תרכובות יש לפי קולטן:**
```sparql
PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?targetName (COUNT(DISTINCT ?c) AS ?n) WHERE {
  ?act odo:hasCompound ?c ; odo:hasTarget ?t .
  ?t odo:targetName ?targetName .
} GROUP BY ?targetName ORDER BY DESC(?n)
```

---

## שלב 7 – אימות הנתונים (אופציונלי)

```bash
conda run -n odo python3 validate_kg.py
```

מריץ 14 שאילתות בדיקה ומדפיס דוח מפורט.

---

## פתרון בעיות נפוצות

| בעיה | פתרון |
|---|---|
| `Connection refused` בהרצת setup | ודא שGraphDB פועל ולחץ Start |
| `Repository already exists` | לא שגיאה – הסקריפט ממשיך בייבוא |
| `ImportError: No module named rdflib` | הרץ: `conda activate odo && pip install rdflib` |
| הייבוא נכשל באמצע | הרץ מחדש – הסקריפט מוסיף על גבי מה שכבר יובא |

---

## Namespace Prefixes

| קיצור | URI |
|---|---|
| `odo:` | `http://odo-project.org/ontology#` |
| `odod:` | `http://odo-project.org/data#` |

ניתן להוסיף אותם ב-GraphDB תחת **Setup → Namespaces**.
