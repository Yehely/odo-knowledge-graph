# מחלץ נתונים אוטומטי ל-ChEMBL — הסבר מקיף

## מה המודל עושה?

המודל (`chembl_fetcher.py`) הוא **מחלץ נתונים אוטומטי** המיועד לאכלוס שדות במסד נתונים פארמקולוגי-אונטולוגי.

הוא מקבל **מזהה מולקולה מ-ChEMBL** (לדוגמה: `CHEMBL101454`) ומייצר שורות Excel מלאות בפורמט של מסד הנתונים — **131 עמודות** המכסות מידע כימי, פרוטאומי, ביולוגי וביבליוגרפי.

---

## מקורות המידע

המודל שולף נתונים מ-**6 מקורות חיצוניים** ומשלב אותם לשורה אחת מאוחדת:

| מקור | מידע שנשלף |
|------|------------|
| **ChEMBL** | פעילות ביולוגית (Ki, IC50, EC50), פרטי assay, מזהה מטרה |
| **PubChem** | SMILES, InChI, InChIKey, שם IUPAC |
| **UniProt** | שם חלבון, GO annotations (תהליך ביולוגי), מזהי PRO |
| **InterPro** | משפחת חלבון, קטגוריה (GPCR וכו'), קיצות G-protein |
| **OLS4 / BTO** | מזהי אונטולוגיה (BTO לרקמות, PRO לחלבונים) |
| **PubMed** | שנת פרסום, כתב עת, DOI |

בנוסף, המודל מפעיל **NLP (עיבוד שפה טבעית)** על תיאור המבחן מ-ChEMBL כדי לחלץ:
- סוג פורמט המבחן (in vitro / in vivo / ex vivo)
- שיטת מדידה (radioligand binding / scintillation counting)
- ליגנד רדיואקטיבי של ייחוס (למשל `[3H]-naloxone`)
- מסלול מתן תרופה (intracerebroventricular / intravenous / וכו')

---

## הגישה הראשונית — ChEMBL בלבד (44.4% דיוק)

הגישה הראשונית שלפה נתונים ישירות ומ-ChEMBL ללא עיבוד נוסף. הבעיות שנמצאו:

### 1. שדות אונטולוגיים חסרים לחלוטין
ChEMBL אינו מספק מזהי BAO (BioAssay Ontology), GO (Gene Ontology), NCIT, PRO, BTO, CLO — כל השדות הללו נשארו ריקים.

### 2. סיווג שגוי של פורמט מבחן
ChEMBL מסווג מבחנים לפי `bao_label`, אך הסיווג שלו אינו תמיד נכון:
- **מבחני radioligand binding** על ממברנות תאיות — ChEMBL מסמן אותם כ-"cell-based" (כי הוא רושם את שם קו התאים), אך בפועל הם **cell-free** (על ממברנה בודדת, לא תא שלם)
- **מבחני in vivo** עם ליגנד רדיואקטיבי — הקוד הראשוני "ראה" את `[3H]` וסיווג כ-cell-free במקום in vivo

### 3. פרטי חלבון חסרים
UniProt וInterPro לא נשאלו, כך שחסרו:
- שם משפחת חלבון (InterPro)
- קישור GO לנתיב סינגול (GPCR signaling pathway)
- מזהי NCIT לתת-משפחה (G(i) Alpha וכו')

### 4. SMILES/InChI לא מלאים
ה-SMILES מ-ChEMBL לעיתים חסר, והמזהה מ-PubChem נשלף בדרך שגויה.

---

## שיפורים לפי גרסאות

### v1 — בסיס ChEMBL בלבד
**דיוק: 44.4%**

שליפה ישירה מ-ChEMBL. רוב השדות האונטולוגיים ריקים.

---

### v2 — שיפורי API
**דיוק: 73.5%**

- הוספת שאילתות ל-PubChem (SMILES, InChI, שם IUPAC)
- הוספת שאילתות ל-UniProt (שם חלבון)
- תיקון בסיסי של מיפוי שדות

---

### v3 — תיקון PubChem CID ו-NCIT
**דיוק: 78.4%**

- תיקון שליפת `pubchem_cid` (מזהה ייחודי ב-PubChem)
- תיקון שדות NCIT (model system, vertebrate taxonomy)

---

### v4 — PRO IDs, GO normalization, InterPro
**דיוק: 80.9%**

- שליפת מזהי PRO דרך OLS4 (Ontology Lookup Service) לפי שם חלבון
- נורמליזציה של GO terms: עדיפות ל-GO:0007193 (adenylate cyclase-inhibiting) על פני GO:0007186 (generic GPCR)
- שליפת קטגוריות InterPro ותיקון הסכמה (ID ושם מוחלפים ב-DB!)
- מיפוי PANTHER → DTO GPCR category

---

### v5c — Cell-free format, Radioligand, Journal
**דיוק: 80.9% (Delta) / 81.0% (Kappa in vitro)**

תיקונים עיקריים:

#### זיהוי cell-free לפני בדיקת ChEMBL
נוסף שלב NLP שרץ **לפני** הבדיקה האם ל-assay יש `cell_chembl_id`. אם התיאור מכיל מילות מפתח כגון "cell membranes", "membrane preparation", "[35S]GTPγS" — המבחן מסווג אוטומטית כ-cell-free, גם אם ChEMBL רשם שם של קו תאים.

#### תבניות מבחן ספציפיות
- **[35S]GTPγS binding**: `bao_bioassay_1 = "G protein activation assay"`, שיטה = radioligand binding, מדידה = scintillation counting
- **Radioligand displacement**: `bao_bioassay_1 = "radioligand binding assay"`, BAO_0002776
- **Subcellular format**: זיהוי "cell-membrane format" (BAO_0000249) כשיש "membranes" בתיאור

#### נורמליזציה של כתבי עת
טבלת מיפוי לשמות כתבי עת (למשל "J Med Chem" → "J. Med. Chem.")

---

### v6c — תמיכה מלאה ב-In Vivo
**דיוק: 80.2% (Delta) / 82.3% (Kappa) / 82.2% (Mu in vivo — שיפור של +25.3%!)**

זוהי הגרסה הנוכחית. השיפור הגדול ביותר הגיע מתמיכה בניסויים in vivo.

#### בעיה שנפתרה: זיהוי in vivo
בגרסאות קודמות, ניסוי in vivo שהכיל `[3H]` בתיאורו (ליגנד רדיואקטיבי) סווג בטעות כ-cell-free. כעת נוסף שלב בעדיפות ראשונה:

> אם התיאור מתחיל ב-"In vivo" → הפורמט הוא tissue-based, הסטינג הוא "in vivo", גם אם יש [3H] בתיאור.

#### שדות חדשים שנוספו עבור in vivo

| שדה | מקור | דוגמה |
|-----|------|--------|
| `bao_experimental_setting` | דגל is_in_vivo | "in vivo" (BAO_0020009) |
| `ncit_model_system` | דגל is_in_vivo | "whole organism" (C77665) |
| `bao_assay_method` | דגל is_in_vivo | "in vivo assay method" (BAO_0000406) |
| `ncit_animal_model` | ChEMBL assay_organism → מיפוי | "guinea pig" |
| `ncit_vertebrate_taxonomy` | ChEMBL assay_organism | "Cavia porcellus" |
| `ncit_vertebrate_taxonomy_id` | ChEMBL assay_tax_id | 10141 |
| `ncit_route_of_administration` | NLP מהתיאור | "intracerebroventricular" |
| `subcellular_format` | היסק: in vivo + radioligand | "tissue-membrane format" (BAO_0020012) |

#### תיקון GO Signaling Pathway
כל הרצפטורים לאופיואידים (Mu, Kappa, Delta) מצמידים דרך Gi → מעכבים adenylate cyclase. UniProt לעיתים מחזיר רק GO:0007186 (generic GPCR). כעת:

> אם `ncit_subfamily == "G(i) Alpha"` → **תמיד** מחזיר GO:0007193 ("adenylate cyclase-inhibiting G protein-coupled receptor signaling pathway"), גם אם UniProt מצא ערך אחר.

#### סטריאוכימיה ב-Parent Structure
`rdkit_parent_structure_smiles/inchi/inchi_key` — כעת שומרים את הגרסה עם סטריאוכימיה מ-ChEMBL, במקום גרסה מנוקה (flat). זה תיקן את הניסויים in vivo שם ה-DB מכיל SMILES עם `@`.

---

## תוצאות סופיות (v6c)

| מולקולה | סוג ניסוי | דיוק |
|---------|-----------|------|
| CHEMBL4172559 (Delta agonist) | Ex vivo, רקמה | **80.2%** |
| CHEMBL101454 (Kappa agonist) | In vitro, cell-free | **82.3%** |
| CHEMBL2021537 (Mu agonist) | **In vivo**, מוח חזיר ניסיוני | **82.2%** |

השיפור הכולל מגישה ראשונית (ChEMBL בלבד):
- Delta: **44.4% → 80.2%** (+35.8%)
- Kappa: **44.4% → 82.3%** (+37.9%)
- Mu in vivo: **44.4% → 82.2%** (+37.8%)

---

---

### v7 — מודלי ML לעמודות שלא ניתן לחלץ ישירות

גרסה זו מוסיפה **חיזוי ממוחשב** לעמודות שאינן מופיעות ב-APIs הרגילים — תוך שימוש בשלוש שיטות שונות.
עמודות אלו מסומנות בצבע שונה בקובץ האקסל (ראה להלן).

#### קטגוריה א׳ — מאפייני ADMET: עמודות qikprop_* (צהוב בקובץ)

תוכנת Schrödinger QikProp היא קניינית ואינה נגישה. במקומה, אומנו מודלים מקומיים:

| מאפיין | ערך | שיטה |
|--------|-----|-------|
| **כלי אימון** | `train_qikprop_models.py` | Random Forest Regressor (scikit-learn) |
| **נתוני אימון** | 12,905 מולקולות מה-DB | שדות qikprop_* מהדאטהבייס |
| **פיצ'רים** | 217 דסקריפטורים מולקולריים | RDKit (Lipinski, TPSA, logP, FCFP, ECFP...) |
| **עמודות שנחזות** | 9 | SASA, FISA, HBA, HBD, QPlogPw, QPlogPo/w, QPlogS, QPlogKhsa, %OralAbs |
| **דיוק (R² על סט בדיקה)** | 0.80–0.97 | SASA 0.95 · HBA/HBD 0.97 · QPlogS 0.80 |
| **לא נחזה** | `qikprop_dipole` | דורש DFT/מכניקה קוונטית |

> **כיצד עובד?** מ-SMILES → RDKit → 217 מספרים → RF מחזיר ערך רציף (רגרסיה).
> כל עמודה = מודל נפרד שמור ב-`nlp_models/qikprop_*.joblib`.

---

#### קטגוריה ב׳ — עמודות אסיי: מסומנות כחול בקובץ

עמודות אלו אינן ב-APIs חיצוניים אך קיים מידע לאימון ב-DB הפנימי.

**1. assay_kit — שם הקיט הניסיוני**

| פרמטר | ערך |
|--------|-----|
| **שיטה** | TF-IDF + Random Forest Classifier |
| **קובץ אימון** | `train_nlp_models.py` |
| **קלט** | `assay_description` (טקסט חופשי) |
| **פלט** | שם קיט (Cisbio, Fluo-4, PathHunter...) — 13 קטגוריות |
| **נתוני אימון** | ~2,678 שורות מה-DB |
| **דיוק (accuracy)** | 98.9% על סט בדיקה |
| **ייצוג טקסט** | TfidfVectorizer(ngram_range=(1,2), max_features=10,000) |
| **מסווג** | RandomForestClassifier(n_estimators=200, random_state=42) |

> **כיצד עובד?** טקסט האסיי → מטריצת TF-IDF bi-gram → RF מחזיר שם קיט.
> מסונן: מחלקות עם פחות מ-10 דוגמאות מוסרות.

---

**2. bao_reference_compound — תרכובת הייחוס**

| פרמטר | ערך |
|--------|-----|
| **שיטה** | חיפוש keyword ב-`assay_description` |
| **מילוניות** | רשימת ~20 תרכובות ידועות (DPDPE, DAMGO, SNC80, U50,488H...) |
| **יישום** | `re.search(pattern, desc, re.IGNORECASE)` |
| **גמישות** | מזהה גם וריאנטים: "U50,488" / "U50488" / "U50 488" |

---

**3. dose_reference_compound — ריכוז תרכובת הייחוס**

| פרמטר | ערך |
|--------|-----|
| **שיטה** | regex למספר + יחידת ריכוז |
| **תבנית** | `(\d+(?:\.\d+)?)\s*(nM\|μM\|uM\|mM\|μg)` |
| **הקשר** | רק בהקשר של מילות ייחוס ("reference", "competing", "displacement") |
| **פלט** | מחרוזת כמו "0.5 nM" או "500 μM" |

---

**4. ncit_route_of_administration_id — נתיב מתן תרופה (NCIT ID)**

| פרמטר | ערך |
|--------|-----|
| **שיטה** | lookup dict סטטי: שם → NCIT ID |
| **מיפוי** | "oral" → C38288, "intravenous" → C38276, "subcutaneous" → C38299, "intraperitoneal" → C38261, "intramuscular" → C28161, "intrathecal" → C38253... |
| **קלט** | `ncit_route_of_administration` (נחלץ בנפרד ע"י NLP) |

---

**5. mi_database_citation_id — מזהה URI של מסד הנתונים**

| פרמטר | ערך |
|--------|-----|
| **שיטה** | lookup dict סטטי: שם → URI |
| **מיפוי** | "ChEMBL Database", "PubMed" → `http://purl.obolibrary.org/obo/MI_0446` |

---

#### סימון בקובץ האקסל

| צבע | משמעות |
|-----|--------|
| **צהוב** | עמודות qikprop_* — הערכה ע"י RF על דסקריפטורים מולקולריים |
| **כחול בהיר** | עמודות assay ML — RF text classifier, regex, או lookup סטטי |
| **ללא צבע** | ערך נחלץ ישירות מ-API / מקור מוסמך |

---

## שדות שנותרו ללא פתרון אוטומטי

| שדה | סיבה |
|-----|------|
| `chembl_chemical_entity_key` | מספור ידני מהמאמר (תרכובת 1, (-)-1 וכו') |
| `ncit_animal_model_strain` | זן בעל חיים (למשל "Dunkin Hartley") — לא ב-ChEMBL |
| `assay_description` (מלא) | פרוטוקול ניסוי מלא — שמור רק ב-DB, לא ב-ChEMBL |
| `odo_assay_property` | תיאור תפוצה ידני ("endpoint after animal sacrificed") |
| `rdkit_parent_structure_*` (חלקית) | הבדלי טאוטומר בין ChEMBL ל-DB (keto vs enol) |
| `qikprop_dipole` | דורש חישוב DFT/מכניקה קוונטית — לא ניתן לאמידה ע"י RF |
| `bao_assay_format_l2/l3` | רק ~40 שורות ב-DB — אין מספיק נתונים לאימון |
| `sem_value`, `cl_lower/upper_95%` | נתוני שונות ניסיוניים — לא ניתנים לחישוב |
| `chembl_binding_site_description` | רק 4 שורות ב-DB |

---

## הרצת המודל

### דרישות מערכת

| רכיב | גרסה מינימלית |
|------|---------------|
| **Python** | 3.9 ומעלה |
| **זיכרון RAM** | 4 GB לפחות (מומלץ 8 GB לאימון) |
| **שטח דיסק** | ~1 GB פנוי (לקבצי המודלים) |
| **גישה לאינטרנט** | נדרשת (שאילתות API חיצוניות) |

---

### שלב 1 — שיבוט ומיקום

```bash
git clone <repo-url>
cd LLM-data/Code/ClaudeAutoExtractor
```

---

### שלב 2 — התקנת תלויות

```bash
pip install -r requirements.txt
```

> מומלץ להשתמש בסביבה וירטואלית:
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

---

### שלב 3 — אימון מודלי ML (חובה לפני ריצה ראשונה)

קבצי המודלים אינם מאוחסנים ב-Git (גודלם עד ~928 MB). יש לאמן אותם מקומית **פעם אחת** לפני השימוש. נתוני האימון כלולים ב-repository תחת `DataBase/`.

#### 3א — מודלי QikProp (מאפיינים פיזיקו-כימיים)

```bash
python3 train_qikprop_models.py
```

- מאמן 9 מודלי Random Forest לחיזוי מאפייני ADMET (SASA, FISA, HBA, HBD, QPlogPw, QPlogPo/w, QPlogS, QPlogKhsa, %OralAbs)
- נתוני אימון: 12,905 מולקולות, 217 דסקריפטורים מולקולריים (RDKit)
- פלט: 9 קבצים בתיקיית `qikprop_models/` (~925 MB)
- זמן ריצה: 10–30 דקות
- דיוק R²: 0.80–0.97

#### 3ב — מודל NLP לסיווג ערכת ניסוי

```bash
python3 train_nlp_models.py
```

- מאמן TF-IDF + Random Forest לחיזוי שדה `assay_kit` (13 קטגוריות)
- פלט: `nlp_models/assay_kit.joblib` (~3 MB)
- זמן ריצה: פחות מדקה
- דיוק: 98.9% על סט בדיקה

> **שים לב:** אם תיקיות המודלים ריקות — המחלץ ירוץ אך עמודות ML יישארו ריקות. יש לוודא שהאימון הושלם לפני הפקת תוצאות.

---

### שלב 4 — הרצת המחלץ

#### אפשרות א׳ — שורת פקודה (CLI)

```bash
# מולקולה בודדת לפי מזהה ChEMBL
python3 chembl_fetcher.py --ids CHEMBL101454 --output outputs/result.xlsx

# מספר מולקולות בבת אחת
python3 chembl_fetcher.py --ids CHEMBL101454 CHEMBL2021537 --output outputs/result.xlsx

# לפי SMILES
python3 chembl_fetcher.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --output outputs/result.xlsx

# מקובץ טקסט (שורה אחת לכל מזהה)
python3 chembl_fetcher.py --file molecules.txt --output outputs/result.xlsx
```

הפלט הוא קובץ Excel בן 131 עמודות בתיקיית `outputs/`.

#### אפשרות ב׳ — ממשק גרפי (Web UI)

```bash
python3 app.py
```

פתחו דפדפן בכתובת: `http://localhost:5050`

הממשק מאפשר הזנת מזהי ChEMBL, מעקב אחר התקדמות החילוץ בזמן אמת, והורדת קובץ האקסל.

---

### שלב 5 — השוואת דיוק מול מסד הנתונים (אופציונלי)

```bash
python3 compare_accuracy.py --output outputs/result.xlsx

# שמירת דוח מפורט
python3 compare_accuracy.py --output outputs/result.xlsx --report outputs/accuracy_report.xlsx
```

הסקריפט משווה כל שדה בין הפלט שנוצר לבין ה-DB ומחשב דיוק לפי שדה ודיוק כולל.

---

### סיכום זרימת העבודה המלאה

```
git clone → pip install → train_qikprop_models.py → train_nlp_models.py
    → chembl_fetcher.py --ids ... → [compare_accuracy.py]
```
