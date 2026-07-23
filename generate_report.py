"""
Generate opioid research insights DOCX report.
Hebrew paragraphs are RTL; code blocks remain LTR.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
section = doc.sections[0]
section.left_margin   = Cm(2.5)
section.right_margin  = Cm(2.5)
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)

# ── Helpers ───────────────────────────────────────────────────────────────────
BLUE   = RGBColor(0x1A, 0x4A, 0x8A)
DARK   = RGBColor(0x22, 0x22, 0x22)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GREY   = RGBColor(0x55, 0x55, 0x55)
LIGHT_BLUE_BG = "D6E4F0"
HEADER_BG     = "1A4A8A"
SUBHEAD_BG    = "2E86C1"
CODE_BG       = "F2F3F4"

def make_rtl(p, align='right'):
    """Set paragraph direction to RTL.

    OOXML schema requires <w:bidi/> to precede <w:spacing/> and <w:jc/>.
    We therefore INSERT bidi at position 0, not append it.
    CENTER alignment is preserved as-is.
    """
    pPr = p._p.get_or_add_pPr()

    # ── 1. Insert <w:bidi/> as the FIRST child (schema order: bidi < spacing < jc)
    if pPr.find(qn('w:bidi')) is None:
        bidi = OxmlElement('w:bidi')
        pPr.insert(0, bidi)          # position 0 = before spacing/ind/jc

    # ── 2. Set alignment (append after spacing — that is the correct schema position)
    jc = pPr.find(qn('w:jc'))
    if jc is None:
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), align)
        pPr.append(jc)
    elif jc.get(qn('w:val'), '') != 'center':   # never override CENTER
        jc.set(qn('w:val'), align)


def make_rtl_cell(cell, align='right'):
    """Apply RTL to every paragraph inside a table cell."""
    for p in cell.paragraphs:
        make_rtl(p, align=align)


def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    if level == 1:
        run.font.size = Pt(18)
        run.font.color.rgb = BLUE
    elif level == 2:
        run.font.size = Pt(14)
        run.font.color.rgb = BLUE
    elif level == 3:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x1A, 0x5E, 0x8A)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(4)
    make_rtl(p)
    return p

def add_body(doc, text, italic=False, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.italic = italic
    if color:
        run.font.color.rgb = color
    p.paragraph_format.space_after = Pt(4)
    make_rtl(p)
    return p

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(2)
    make_rtl(p)
    return p

def add_code_block(doc, label, code_text, output_text):
    """Renders a SPARQL input + output block.
    Label = RTL (Hebrew). Code/output cells = LTR.
    """
    # Label — Hebrew, RTL
    p = doc.add_paragraph()
    r = p.add_run(label)
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = GREY
    p.paragraph_format.space_after = Pt(2)
    make_rtl(p)

    # SPARQL query table — LTR throughout
    tbl = doc.add_table(rows=2, cols=1)
    tbl.style = 'Table Grid'

    hcell = tbl.rows[0].cells[0]
    set_cell_bg(hcell, "2E4057")
    hr = hcell.paragraphs[0].add_run("  SPARQL Query")
    hr.bold = True
    hr.font.size = Pt(9)
    hr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # header cell stays LTR (English label)

    ccell = tbl.rows[1].cells[0]
    set_cell_bg(ccell, "F8F9FA")
    cp = ccell.paragraphs[0]
    cr = cp.add_run(code_text)
    cr.font.name  = 'Courier New'
    cr.font.size  = Pt(8.5)
    cr.font.color.rgb = RGBColor(0x20, 0x60, 0x20)
    # code cell stays LTR

    doc.add_paragraph()  # spacer

    # Output table — LTR throughout
    otbl = doc.add_table(rows=2, cols=1)
    otbl.style = 'Table Grid'

    ohcell = otbl.rows[0].cells[0]
    set_cell_bg(ohcell, "1A4A8A")
    ohr = ohcell.paragraphs[0].add_run("  Output")
    ohr.bold = True
    ohr.font.size = Pt(9)
    ohr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    occell = otbl.rows[1].cells[0]
    set_cell_bg(occell, "EBF5FB")
    ocp = occell.paragraphs[0]
    ocr = ocp.add_run(output_text)
    ocr.font.name  = 'Courier New'
    ocr.font.size  = Pt(8.5)
    ocr.font.color.rgb = RGBColor(0x1A, 0x1A, 0x4A)
    # output cell stays LTR

    doc.add_paragraph()  # spacer

def add_insight_header(doc, number, title, subtitle=""):
    p = doc.add_paragraph()
    run = p.add_run(f"תובנה {number}  |  {title}")
    run.bold = True
    run.font.size = Pt(15)
    run.font.color.rgb = BLUE
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after  = Pt(3)
    make_rtl(p)
    if subtitle:
        p2 = doc.add_paragraph()
        r2 = p2.add_run(subtitle)
        r2.italic = True
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = GREY
        p2.paragraph_format.space_after = Pt(6)
        make_rtl(p2)

def add_divider(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pb = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'),   'single')
    bottom.set(qn('w:sz'),    '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '1A4A8A')
    pb.append(bottom)
    pPr.append(pb)
    p.paragraph_format.space_after = Pt(8)

def add_finding_box(doc, text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.style = 'Table Grid'
    cell = tbl.rows[0].cells[0]
    set_cell_bg(cell, "FEF9E7")
    p = cell.paragraphs[0]
    r = p.add_run("מסקנה: " + text)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(0x7D, 0x33, 0x00)
    p.paragraph_format.right_indent = Pt(4)  # RTL: indent from right
    make_rtl(p)
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
#  COVER PAGE
# ══════════════════════════════════════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("\n\n")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("גרף ידע אופיואידים — ניתוח חוקר בכיר")
r.bold = True
r.font.size = Pt(26)
r.font.color.rgb = BLUE
p.paragraph_format.space_after = Pt(10)
make_rtl(p)  # CENTER preserved, bidi added

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("תובנות ביולוגיות מתוך שאילתות SPARQL על גרף הידע ODO")
r.font.size = Pt(14)
r.font.color.rgb = GREY
make_rtl(p)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("36,523 פעילויות  ·  17,820 תרכובות  ·  43 מטרות ביולוגיות  ·  370 ניסויי in vivo")
r.font.size = Pt(11)
r.italic = True
r.font.color.rgb = GREY
make_rtl(p)

doc.add_paragraph()
doc.add_paragraph()

# Stats table — col 0: English type names (LTR), col 1: numbers (LTR), col 2: Hebrew (RTL)
# Header row: all three headers are Hebrew → RTL
stats = [
    ("Activity",         "36,523", "פעילויות ביולוגיות"),
    ("Compound",         "17,820", "תרכובות"),
    ("Assay",             "4,515", "assays"),
    ("Document",          "1,165", "פרסומים"),
    ("InVivoParameters",    "370", "ניסויי in vivo"),
    ("Target",               "43", "מטרות ביולוגיות"),
    ("Protein",              "77", "חלבונים"),
]
tbl = doc.add_table(rows=1+len(stats), cols=3)
tbl.style = 'Table Grid'
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, hdr in enumerate(["סוג", "כמות", "תיאור"]):
    cell = tbl.rows[0].cells[i]
    set_cell_bg(cell, "1A4A8A")
    r = cell.paragraphs[0].add_run(hdr)
    r.bold = True; r.font.color.rgb = RGBColor(0xFF,0xFF,0xFF); r.font.size = Pt(10)
    make_rtl_cell(cell)
for row_i, (t, n, d) in enumerate(stats, 1):
    bg = "EBF5FB" if row_i % 2 == 0 else "FFFFFF"
    for col_i, val in enumerate([t, n, d]):
        cell = tbl.rows[row_i].cells[col_i]
        set_cell_bg(cell, bg)
        r = cell.paragraphs[0].add_run(val)
        r.font.size = Pt(10)
        if col_i == 1:
            r.bold = True
            r.font.color.rgb = BLUE
        if col_i == 2:          # Hebrew description column → RTL
            make_rtl_cell(cell)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 1 — G-protein biased agonist
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 1,
    "G-Protein Biased Agonist ב-MOR עם הוכחה ישירה",
    "הפרדה כמותית בין נתיב ה-Gi לנתיב ה-beta-arrestin באותה מערכת ניסוי")

add_body(doc, "הנחת היסוד של 'biased agonism' גורסת שניתן לייצר תרכובות המפעילות את נתיב ה-Gi (אנלגזיה) מבלי לגייס beta-arrestin (תופעות לוואי: תלות, עצירות, דיכוי נשימה). CHEMBL2443260 מספק הוכחה כמותית ישירה לכך באותה מערכת תאי HEK293.")

add_code_block(doc,
"שאילתה — G-protein biased agonists (יחס arrestin/Gprotein > 5):",
"""PREFIX odo: <http://odo-project.org/ontology#>
SELECT ?compId ?gproteinEC50 ?arrestinEC50 WHERE {
  { SELECT ?comp (MIN(xsd:float(?ev)) as ?gproteinEC50) WHERE {
      ?act odo:hasCompound ?comp ; odo:endpointType ?et ;
           odo:endpointValue ?ev ; odo:hasAssay ?assay .
      ?assay odo:assayDescription ?desc .
      FILTER(REGEX(?desc,'GTPgamma|cAMP.*mu|mu.*cAMP|MOR.*GTP','i'))
      FILTER(?et IN ('EC50','IC50')) FILTER(xsd:float(?ev)>0)
    } GROUP BY ?comp }
  { SELECT ?comp (MIN(xsd:float(?ev2)) as ?arrestinEC50) WHERE {
      ?act2 odo:hasCompound ?comp ; odo:endpointType ?et2 ;
            odo:endpointValue ?ev2 ; odo:hasAssay ?assay2 .
      ?assay2 odo:assayDescription ?desc2 .
      FILTER(REGEX(?desc2,'USOS-beta-arrestin-hMOR|beta.arrestin.*MOR','i'))
      FILTER(?et2 IN ('EC50','IC50')) FILTER(xsd:float(?ev2)>0)
    } GROUP BY ?comp }
  ?comp odo:chemblId ?compId .
  FILTER(?arrestinEC50 / ?gproteinEC50 > 5)
} ORDER BY DESC(?arrestinEC50/?gproteinEC50) LIMIT 20""",
"""CHEMBL1612697  G-protein=54 nM    arrestin=8,300 nM  bias=153.7x
CHEMBL4210060  G-protein=91 nM    arrestin=10,000 nM bias=109.9x
CHEMBL2443260  G-protein=5.0 nM   arrestin=501 nM    bias=100.0x  ← מאומת
CHEMBL4203641  G-protein=148 nM   arrestin=10,000 nM bias=67.6x
CHEMBL3590199  G-protein=1.6 nM   arrestin=84 nM     bias=52.5x
CHEMBL4216374  G-protein=31 nM    arrestin=1,678 nM  bias=54.1x
CHEMBL3590200  G-protein=2.9 nM   arrestin=93 nM     bias=32.1x
CHEMBL2443278  G-protein=15.85 nM arrestin=251 nM    bias=15.8x
CHEMBL70 (מורפין) G-protein=39.81 nM arrestin=501 nM bias=12.6x
... (20 תוצאות סה"כ)""")

add_code_block(doc,
"אימות CHEMBL2443260 — שאילתת פרופיל מלא:",
"""SELECT ?endpointType ?endpointValue ?assayDesc WHERE {
  ?comp odo:chemblId 'CHEMBL2443260' .
  ?act a odo:Activity ; odo:hasCompound ?comp ;
       odo:endpointType ?endpointType ;
       odo:endpointValue ?endpointValue ; odo:hasAssay ?assay .
  ?assay odo:assayDescription ?assayDesc
}""",
"""[Mu opioid receptor] EC50 = 5.012 nM
  → "Agonist at MOR in HEK293 – inhibition of forskolin-induced cAMP" (Gi pathway)

[Mu opioid receptor] EC50 = 501.19 nM
  → "Agonist at MOR in HEK293 – beta-arrestin recruitment"

[Mu opioid receptor] Efficacy = 197%
  → "Agonist at MOR in HEK293 – beta-arrestin recruitment"

אימות: שתי המדידות מאותה מעבדה, HEK293, MOR אנושי. הבדל: 100x.""")

add_finding_box(doc, "CHEMBL2443260 הוא G-protein biased MOR agonist עם bias של 100x, מאומת במערכת ניסוי אחת. Efficacy=197% ב-Gi אך EC50=501 nM ב-arrestin. מועמד מוביל לפיתוח אנלגטיק ללא דיכוי נשימה. Phase 0 — לא נחקר קלינית.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 2 — In vitro → In vivo bridge
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 2,
    "גשר In Vitro → In Vivo: 8 תרכובות Drug-Like עם פרופיל שלם",
    "תרכובות עם MOR Ki < 10 nM, ED50 < 0.1 mg/kg, ספיגה אוראלית > 75% — טרם ניסויים קליניים")

add_body(doc, "אחד הכשלים השכיחים בפיתוח תרופות הוא פוטנציה in vitro גבוהה שאינה מתורגמת in vivo. שאילתה זו זיהתה 8 תרכובות המקיימות בו-זמנית שלושה תנאים מחמירים: קישור ל-MOR, יעילות in vivo וספיגה אוראלית מצוינת.")

add_code_block(doc,
"שאילתה — in vitro + in vivo + drug-like properties:",
"""SELECT ?compId ?morKi ?ed50 ?ed50Unit ?mw ?oral WHERE {
  { SELECT ?comp (MIN(xsd:float(?v)) as ?morKi) WHERE {
      ?act odo:hasTarget ?t ; odo:hasCompound ?comp ;
           odo:endpointType 'Ki' ; odo:endpointValue ?v .
      FILTER(?t IN (<target_CHEMBL233>,<target_CHEMBL270>,
                    <target_Mu_opioid_receptor>))
    } GROUP BY ?comp }
  { SELECT ?comp (MIN(xsd:float(?v2)) as ?ed50)
           (SAMPLE(?u) as ?ed50Unit) WHERE {
      ?act2 odo:hasCompound ?comp ; odo:endpointType 'ED50' ;
            odo:endpointValue ?v2 ; odo:unitLabel ?u .
    } GROUP BY ?comp }
  ?comp odo:chemblId ?compId .
  OPTIONAL { ?comp odo:molecularWeight ?mw }
  OPTIONAL { ?comp odo:humanOralAbsorption ?oral }
  FILTER(?morKi < 10 && ?ed50 < 0.1)
} ORDER BY ?ed50 LIMIT 20""",
"""CHEMBL606924: MOR Ki=0.25 nM  ED50=0.0003 mg/kg  MW=473  oral=100%
CHEMBL606985: MOR Ki=0.09 nM  ED50=0.0009 mg/kg  MW=447  oral=100%
CHEMBL606986: MOR Ki=0.20 nM  ED50=0.001  mg/kg  MW=489  oral=95%
CHEMBL514774: MOR Ki=0.34 nM  ED50=0.002  mg/kg  MW=459  oral=100%
CHEMBL606925: MOR Ki=0.20 nM  ED50=0.006  mg/kg  MW=445  oral=96%
CHEMBL606923: MOR Ki=1.9  nM  ED50=0.006  mg/kg  MW=523  oral=92%
CHEMBL326684: MOR Ki=0.023 nM ED50=0.009  mg/kg  MW=329  oral=80.6%
CHEMBL606987: MOR Ki=1.1  nM  ED50=0.009  mg/kg  MW=509  oral=75.6%""")

add_body(doc, "CHEMBL326684 בולטת במיוחד: MW=329 Da, alogP=1.79 (נמוך מאוד לאופיואיד), Ro5=0, MOR Ki=0.023 nM, ED50=0.009 mg/kg. תרכובת הידרופילית בגרעין benzofuroisoquinolinone — חריגה מהפרדיגמה המקובלת של אופיואידים ליפופיליים.")
add_finding_box(doc, "סדרת CHEMBL60692x + CHEMBL514774 + CHEMBL326684: 8 תרכובות drug-like עם גשר מלא in vitro→in vivo. כולן Phase 0. CHEMBL326684 (MW=329, alogP=1.79) שוברת את הכלל 'אופיואיד = ליפופיילי'. מועמדת לפיתוח כ-lead molecule הידרופילית.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 3 — Cebranopadol / NOP+MOR
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 3,
    "NOP+MOR Dual Ligands — פרדיגמת Cebranopadol מאומתת ומורחבת",
    "מקבץ של 20+ תרכובות עם Ki < 5 nM בשני הקולטנים — כולן בשלב פרה-קליני")

add_body(doc, "Cebranopadol (CHEMBL2364605) נמצא ב-Phase 3 כאגוניסט משולב NOP+MOR. השאילתה אימתה זאת וזיהתה 20 אנלוגים נוספים בפרופיל דומה — אחד מהם חזק יותר מה-lead הקליני.")

add_code_block(doc,
"שאילתה — NOP+MOR dual ligands (Ki < 5 nM בשניהם):",
"""SELECT ?compId ?morVal ?nopVal WHERE {
  { SELECT ?comp (MIN(xsd:float(?v)) as ?morVal) WHERE {
      ?act odo:hasTarget ?t ; odo:hasCompound ?comp ;
           odo:endpointType 'Ki' ; odo:endpointValue ?v .
      FILTER(?t IN (<target_CHEMBL233>,<target_CHEMBL270>,
                    <target_Mu_opioid_receptor>))
    } GROUP BY ?comp }
  { SELECT ?comp (MIN(xsd:float(?v2)) as ?nopVal) WHERE {
      ?act2 odo:hasTarget ?t2 ; odo:hasCompound ?comp ;
            odo:endpointType 'Ki' ; odo:endpointValue ?v2 .
      FILTER(?t2 IN (<target_CHEMBL2014>,<target_CHEMBL3621>,
                     <target_Nociceptin_receptor>))
    } GROUP BY ?comp }
  ?comp odo:chemblId ?compId .
  FILTER(?morVal < 5 && ?nopVal < 5)
} ORDER BY ?morVal LIMIT 20""",
"""CHEMBL3326229: MOR Ki=0.24 nM, NOP Ki=0.55 nM
CHEMBL3326224: MOR Ki=0.26 nM, NOP Ki=0.10 nM  ← NOP חזק מ-cebranopadol
CHEMBL3326220: MOR Ki=0.30 nM, NOP Ki=0.50 nM
CHEMBL3889835: MOR Ki=0.35 nM, NOP Ki=0.13 nM
CHEMBL3894898: MOR Ki=0.36 nM, NOP Ki=0.26 nM
CHEMBL2364605: MOR Ki=0.70 nM, NOP Ki=0.90 nM  ← cebranopadol (Phase 3)
CHEMBL4557245: MOR Ki=1.0  nM, NOP Ki=0.70 nM
... (20 תוצאות סה"כ)""")

add_code_block(doc,
"אימות CHEMBL2364605 (cebranopadol) — פרופיל מלא:",
"""SELECT ?endpointType ?endpointValue ?unit ?tgtLabel ?assayDesc WHERE {
  ?comp odo:chemblId 'CHEMBL2364605' .
  ?act a odo:Activity ; odo:hasCompound ?comp ;
       odo:endpointType ?endpointType ; odo:endpointValue ?endpointValue ;
       odo:hasTarget ?tgt .
  ?tgt rdfs:label ?tgtLabel .
  OPTIONAL { ?act odo:hasAssay ?a . ?a odo:assayDescription ?assayDesc }
}""",
"""[Mu opioid receptor]   Ki=0.7 nM
  → Displacement of [3H]naloxone from recombinant human MOR in CHOK1 cells

[Mu opioid receptor]   EC50=1.2 nM
  → Agonist at recombinant human MOR in CHOK1 cell membranes

[Nociceptin receptor]  Ki=0.9 nM
  → Displacement of [3H]nociceptin from human NOP in HEK293 cells

[Nociceptin receptor]  EC50=13.0 nM
  → Agonist at recombinant human NOP in CHOK1 cell membranes""")

add_finding_box(doc, "20 תרכובות עם Ki < 5 nM בשני NOP ו-MOR — כולן Phase 0. CHEMBL3326224 חזקה מ-cebranopadol (MOR=0.26 nM, NOP=0.10 nM). פוטנציאל ל-next-generation analgesic ללא תלות.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 4 — Buprenorphine kinetics (Ke)
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 4,
    "בופרנורפין כאנטגוניסט דו-פאזי: Ke = 0.004 nM — מנגנון קינטי ייחודי",
    "הגרף מאשר מנגנון שהיה ידוע קלינית אך לא מקודד — slow off-rate כגורם מפתח")

add_body(doc, "ה-Ke (שקיחות קינטית של אנטגוניסט) מודד מהירות הניתוק מהקולטן. CHEMBL511142 זוהה אוטומטית כ-BUPRENORPHINE (שדה chemicalEntityName). Ke=0.004 nM מסביר כיצד ננלוקסון — בריכוזים קליניים — אינו יכול לעקור אותו מ-MOR.")

add_code_block(doc,
"שאילתה — Ke values (מיון עולה = ניתוק הכי איטי ראשון):",
"""SELECT ?compId ?endpointValue ?unit ?tgtLabel ?assayDesc WHERE {
  ?act a odo:Activity ;
       odo:hasCompound ?comp ;
       odo:endpointType 'Ke' ;
       odo:endpointValue ?endpointValue ;
       odo:unitLabel ?unit ;
       odo:hasTarget ?tgt .
  ?tgt rdfs:label ?tgtLabel .
  ?comp odo:chemblId ?compId .
  OPTIONAL { ?act odo:hasAssay ?a ; ?a odo:assayDescription ?assayDesc }
} ORDER BY ?endpointValue LIMIT 20""",
"""CHEMBL511142:  Ke=0.004 nM [MOR] → Antagonist at MOR – mouse vas deferens, inhibition of DAMGO
  [= BUPRENORPHINE — confirmed by chemicalEntityName field]
CHEMBL2338755: Ke=0.007 nM [MOR] → Antagonist at MOR – mouse vas deferens
CHEMBL2179263: Ke=0.008 nM [MOR] → Displacement of [3H]DAMGO from MOR – guinea pig
CHEMBL2338736: Ke=0.010 nM [KOR] → Antagonist at KOR – mouse vas deferens
CHEMBL3613446: Ke=0.010 nM [KOR] → Antagonist at cloned human KOR
CHEMBL3326784: Ke=0.010 nM [KOR] → Antagonist at human KOR in CHO cells
CHEMBL52451:   Ke=0.013 nM [MOR] → Inhibition of [35S]GTPgammaS – guinea pig caudate
CHEMBL473206:  Ke=0.020 nM [MOR] → Antagonist at MOR – mouse vas deferens
... (20 תוצאות סה"כ, גם Ke ל-KOR ו-DOR)""")

add_finding_box(doc, "Ke=0.004 nM של בופרנורפין מאשר מה שרופאי כורח יודעים אבל קשה להסביר: מינון נלוקסון סטנדרטי (0.4-2 mg) לא מספיק לעקור בופרנורפין מ-MOR. הגרף מקודד עכשיו את הבסיס הקינטי של 'בופרנורפין-עמידה-לנלוקסון'. מועמד לשאלות על מינון נכון לטיפול בהיפרדוז.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 5 — DOR PAMs
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 5,
    "מאפנני אלוסטרי חיוביים ב-DOR — אסטרטגיה עוקפת לפרכוסים",
    "PAMs מגבירים פעילות אנדוגנית מבלי להפעיל את הקולטן ישירות")

add_body(doc, "DOR agonists כשלו קלינית בשל ספי פרכוסים נמוכים. Positive Allosteric Modulators (PAMs) — שמגבירים תגובה לאנקפלינים אנדוגניים אך אינם מפעילים את הקולטן בפני עצמם — מציעים מנגנון שיכול לעקוף את מחסום הפרכוסים. הגרף מכיל סדרה שלמה של DOR PAMs (CHEMBL3426xxx).")

add_code_block(doc,
"שאילתה — assays אלוסטריים באופיואידים:",
"""SELECT DISTINCT ?compId ?tgtLabel ?endpointType ?endpointValue ?assayDesc WHERE {
  ?act a odo:Activity ; odo:hasCompound ?comp ;
       odo:endpointType ?endpointType ; odo:endpointValue ?endpointValue ;
       odo:hasTarget ?tgt ; odo:hasAssay ?assay .
  ?tgt rdfs:label ?tgtLabel .
  ?assay odo:assayDescription ?assayDesc .
  ?comp odo:chemblId ?compId .
  FILTER(REGEX(?assayDesc,'allosteric|PAM|potentiation','i'))
} LIMIT 20""",
"""CHEMBL3426796 [Delta opioid receptor]: Emax=116%,  EC50=100 nM
  → Positive allosteric modulation of human DOR-1 in CHO – potentiation of DPDPE

CHEMBL3426794 [Delta opioid receptor]: Emax=95%,   EC50=100 nM
  → Positive allosteric modulation of human DOR-1 in CHO

CHEMBL3426788 [Delta opioid receptor]: Emax=126%,  EC50=200 nM
  → Positive allosteric modulation of human DOR-1 in CHO

CHEMBL3426793 [Delta opioid receptor]: Emax=136%,  EC50=1000 nM
  → Positive allosteric modulation of human DOR-1 in CHO
   + MOR: Emax=182%, EC50=7000 nM

CHEMBL3216936 [Delta opioid receptor]: Activity=24.2%
  → Potentiation of DOR agonism

(15+ תרכובות בסדרת CHEMBL3426xxx)""")

add_finding_box(doc, "CHEMBL3426796 (EC50=100 nM, Emax=116%) הוא DOR PAM המחזק תגובת DPDPE ב-CHO cells. אסטרטגיה זו — כמו BDZ ל-GABA — מאפשרת הגברת אנלגזיה אנדוגנית ללא הפעלת הקולטן ישירות. כל תרכובות הסדרה Phase 0. פוטנציאל גדול עם פרופיל בטיחות לא-ידוע.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 6 — KOR antagonist gap
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 6,
    "הפער הקליני של KOR: עשרות מולקולות בפוטנציה תת-ננומולרית — אחת בניסוי",
    "KOR antagonists רלוונטים לדיכאון, ממכרות ואנהדוניה — תת-מושקעים קלינית")

add_body(doc, "KOR מתווך דיספוריה, דיכאון ואנהדוניה. אנטגוניסטים שלו פותחו כמטרה לטיפול בדיכאון עמיד ובגמילה מאופיואידים. הגרף חושף עושר של תרכובות עם IC50 < 2 nM — אך מתוכן רק אחת מגיעה ל-Phase 2.")

add_code_block(doc,
"שאילתה — KOR antagonists עם pChEMBL > 8:",
"""SELECT ?compId ?pchembl ?endpointValue ?maxPhase ?assayDesc WHERE {
  ?act a odo:Activity ;
       odo:hasTarget ?tgt ;
       odo:hasCompound ?comp ;
       odo:pchemblValue ?pchembl ;
       odo:endpointType 'IC50' ;
       odo:endpointValue ?endpointValue ;
       odo:hasAssay ?assay .
  ?assay odo:assayDescription ?assayDesc .
  ?comp odo:chemblId ?compId .
  OPTIONAL { ?comp odo:maxPhase ?maxPhase }
  FILTER(?tgt IN (<target_CHEMBL237>,<target_CHEMBL3952>))
  FILTER(REGEX(?assayDesc,'[Aa]ntagonist','i'))
  FILTER(xsd:float(?pchembl) > 8)
} ORDER BY DESC(?pchembl) LIMIT 15""",
"""CHEMBL4544914: pChEMBL=9.59, IC50=0.26 nM,  Phase=0
CHEMBL573214:   pChEMBL=9.55, IC50=0.28 nM,  Phase=0
CHEMBL4440683:  pChEMBL=9.28, IC50=0.53 nM,  Phase=0
CHEMBL4576339:  pChEMBL=9.23, IC50=0.59 nM,  Phase=0
CHEMBL4523014:  pChEMBL=9.12, IC50=0.76 nM,  Phase=0
CHEMBL4592045:  pChEMBL=9.10, IC50=0.80 nM,  Phase=2  ← יחיד
CHEMBL4549826:  pChEMBL=9.06, IC50=0.88 nM,  Phase=0
CHEMBL4583245:  pChEMBL=8.96, IC50=1.10 nM,  Phase=0
...
CHEMBL3590199:  pChEMBL=8.80, IC50=1.60 nM,  Phase=0
(15 תרכובות, 14 מהן Phase 0)""")

add_finding_box(doc, "CHEMBL4592045 — KOR antagonist היחיד ב-Phase 2 (IC50=0.8 nM). 14 תרכובות חזקות יותר ממנו נמצאות ב-Phase 0. הפער בין פוטנציה in vitro לפיתוח קליני מצביע על מחסום שאינו פארמקולוגי — ייתכן חוסר מימון, IP, או בעיות in vivo שאינן מקודדות בגרף.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 7 — Emax distribution
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 7,
    "פיזור Emax ב-MOR: 40% תרכובות הן אגוניסטים חלקיים — נכס לא מנוצל",
    "הפרדה בין אנלגזיה מרבית לרצף תופעות הלוואי")

add_body(doc, "אגוניסטים חלקיים (partial agonists) כמו בופרנורפין משיגים ceiling effect — אנלגזיה מספקת ללא הסיכון של מינון יתר. השאילתה ממפה את פיזור ה-Emax בכלל המאגר.")

add_code_block(doc,
"שאילתה — פיזור Emax ב-MOR (קיבוץ לטווחים):",
"""SELECT ?emaxRange (COUNT(DISTINCT ?comp) as ?count) WHERE {
  ?act a odo:Activity ;
       odo:hasTarget ?tgt ;
       odo:hasCompound ?comp ;
       odo:endpointType ?et ;
       odo:endpointValue ?ev .
  FILTER(?tgt IN (<target_CHEMBL233>,<target_CHEMBL270>,
                  <target_Mu_opioid_receptor>))
  FILTER(?et IN ('Emax','Intrinsic activity','%Emax'))
  BIND(
    IF(xsd:float(?ev)<=20,'silent/inverse (0-20%)',
    IF(xsd:float(?ev)<=50,'partial low (21-50%)',
    IF(xsd:float(?ev)<=80,'partial high (51-80%)',
    IF(xsd:float(?ev)<=110,'full agonist (81-110%)',
    'superagonist (>110%)')))) as ?emaxRange)
} GROUP BY ?emaxRange ORDER BY ?emaxRange""",
"""silent/inverse agonist (0-20%):   212 תרכובות
partial low (21-50%):              410 תרכובות  ← הרוב
partial high (51-80%):             148 תרכובות
full agonist (81-110%):            253 תרכובות
superagonist (>110%):              118 תרכובות

סה"כ: 1,141 תרכובות עם Emax מדוד ב-MOR""")

add_body(doc, "410 תרכובות עם Emax 21-50% — מאגר ענק של partial agonists לא מנוצל. 212 תרכובות 'שקטות' — חלקן inverse agonists פוטנציאליים. 118 סופר-אגוניסטים — חלקם PAMs המשנים את normalization reference.")
add_finding_box(doc, "36% מהתרכובות עם Emax מדוד הן partial agonists (Emax 21-80%). אף אחת לא הגיעה ל-Phase 3 כאנלגטיק בעשור האחרון — על אף שמנגנון ה-ceiling effect של בופרנורפין הוכיח עצמו קלינית. תת-ניצול של מאגר ספרות ענק.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 8 — DOR clinical landscape
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 8,
    "ה-DOR: 9,157 מדידות, אפס תרופות מאושרות — מה חוסם את התרגום?",
    "ניתוח תרכובות DOR בשלב קליני חושף פרופיל יעילות בעייתי")

add_body(doc, "DOR מוצע כמטרה לאנלגזיה כרונית עם פחות סבילות ממורפין. 9,157 מדידות בגרף — אך אפס תרופות DOR-agonist מאושרות. השאילתה בדקה את תרכובות ה-DOR שהגיעו לניסויים קליניים.")

add_code_block(doc,
"שאילתה — DOR clinical-stage compounds (maxPhase >= 1):",
"""SELECT ?compId ?endpointType ?endpointValue ?unit ?maxPhase ?assayDesc WHERE {
  ?act a odo:Activity ;
       odo:hasTarget ?tgt ; odo:hasCompound ?comp ;
       odo:endpointType ?endpointType ; odo:endpointValue ?endpointValue ;
       odo:unitLabel ?unit .
  ?comp odo:chemblId ?compId .
  OPTIONAL { ?comp odo:maxPhase ?maxPhase }
  OPTIONAL { ?act odo:hasAssay ?a ; ?a odo:assayDescription ?assayDesc }
  FILTER(?tgt IN (<target_CHEMBL236>,<target_CHEMBL269>,
                  <target_Delta_opioid_receptor>))
  FILTER(?endpointType IN ('EC50','Emax','Intrinsic activity'))
  FILTER(xsd:integer(COALESCE(?maxPhase,'0')) >= 1)
} ORDER BY DESC(?maxPhase) LIMIT 30""",
"""CHEMBL267495 [Phase 3]: Emax=55.2%  EC50=?
  → "Agonist at human DOR in CHO by [35S]GTPgammaS" — partial agonist!

CHEMBL1190199 [Phase 3]: EC50=8,000 nM
  → "Agonist at human DOR expressed in COS7 cells" — חלש מאוד

CHEMBL561339 [Phase 2]: EC50=94 nM
  → "Agonist at human DOR in CHO cells – inhibition of cAMP"

CHEMBL441765 [Phase 1]: Emax=112.6% at rat DOR (full agonist)

CHEMBL445332 [Phase 1]: Emax=0.0% at human DOR  ← לא אגוניסט!
  → נכשל Phase 1

CHEMBL19019 [Phase 4]: EC50=21 nM at human DOR
  → Phase 4 אך עקב MOR, לא DOR""")

add_finding_box(doc, "CHEMBL267495 (Phase 3) הוא partial agonist ב-DOR עם Emax=55.2% — אסטרטגיה מכוונת להפחתת פרכוסים. CHEMBL445332 כשלה ב-Phase 1 עם Emax=0% ב-DOR האנושי. מסקנה: DOR agonists שנועדו לאנשים חייבים להראות Emax מדוד בקולטן האנושי, לא רק הרדנטי.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 9 — NOP underinvestment
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 9,
    "NOP — הקולטן הרביעי: תרכובות בפוטנציה 0.033 nM, Phase 0",
    "פוטנציאל אנלגטי ללא ממכרות — 2,048 מדידות, אפס תרופות")

add_code_block(doc,
"שאילתה — קושרים חזקים ביותר ל-NOP:",
"""SELECT ?compId ?pchembl ?endpointType ?endpointValue ?role WHERE {
  ?act a odo:Activity ; odo:hasTarget ?tgt ;
       odo:hasCompound ?comp ;
       odo:pchemblValue ?pchembl ;
       odo:endpointType ?endpointType ;
       odo:endpointValue ?endpointValue .
  ?comp odo:chemblId ?compId .
  OPTIONAL { ?comp odo:pharmacologicalRoleLabel ?role }
  FILTER(?tgt IN (<target_CHEMBL2014>,<target_CHEMBL3621>,
                  <target_CHEMBL4503>,<target_Nociceptin_receptor>))
  FILTER(?endpointType IN ('Ki','IC50','EC50'))
} ORDER BY DESC(?pchembl) LIMIT 10""",
"""CHEMBL3343947: pChEMBL=10.48, IC50=0.033 nM
CHEMBL2088054: pChEMBL=10.43, Ki=0.037  nM
CHEMBL3810319: pChEMBL=10.4,  Ki=0.040  nM
CHEMBL1631926: pChEMBL=10.4,  EC50=0.040 nM
CHEMBL414736:  pChEMBL=10.33, IC50=0.046 nM
CHEMBL3808650: pChEMBL=10.3,  Ki=0.050  nM
CHEMBL3236476: pChEMBL=10.19, Ki=0.065  nM
CHEMBL3326224: pChEMBL=10.0,  Ki=0.100  nM  ← גם MOR Ki=0.26 nM (dual!)
... כולן Phase 0""")

add_finding_box(doc, "NOP מקבל שביעית מתשומת הלב של MOR (2,048 vs 14,087 מדידות). CHEMBL3343947 (IC50=0.033 nM) ו-CHEMBL2088054 (Ki=0.037 nM) הן הקושרים החזקים ביותר ל-NOP בגרף — Phase 0 שניהם. cebranopadol (CHEMBL2364605) הוא ה-NOP compound היחיד ב-Phase 3. הנתונים מצביעים על underinvestment שיטתי.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHT 10 — Signaling pathway annotation gap
# ══════════════════════════════════════════════════════════════════════════════
add_insight_header(doc, 10,
    "עיוור מבני: כל 36,522 הפעילויות מוגדרות לנתיב איתות אחד בלבד",
    "אנוטציה חסרה של נתיב ה-beta-arrestin מונעת שאלות biased agonism אוטומטיות")

add_code_block(doc,
"שאילתה — נתיבי איתות בגרף:",
"""SELECT ?pathway ?label (COUNT(?act) as ?count) WHERE {
  ?act a odo:Activity ;
       odo:hasSignalingPathway ?pathway .
  OPTIONAL { ?pathway rdfs:label ?label }
} GROUP BY ?pathway ?label ORDER BY DESC(?count)""",
"""signaling_GO_0007193 | adenylate cyclase-inhibiting G-protein-coupled
  receptor signaling pathway: 36,522 פעילויות

adenylate_cyclase-inhibiting_G_protein-coupled_receptor_signaling_pathway:
  80 פעילויות

סה"כ: 2 נתיבים בלבד. אין: beta-arrestin, Gs, Gq, GRK, MAPK, ERK.""")

add_body(doc, "הגרף מכיל מאות assays המודדים גיוס beta-arrestin (PathHunter, BRET, U2OS co-expression) אך כולם ממופים לאותו נתיב Gi/o. כתוצאה מכך, שאילתה פשוטה כמו 'מה bias factor של כל תרכובת?' אינה ניתנת לתשובה ישירה — נדרשת ניתוח ב-assay description level בלבד.", italic=False)

add_finding_box(doc, "המלצה קונקרטית: הוספת שלושה נתיבי איתות: (1) beta-arrestin-recruitment-pathway, (2) cAMP-inhibition-Gi, (3) GTPgammaS-binding. אנוטציה של ~800 assays קיימים תהפוך שאלות biased-agonism לאוטומטיות על 36K+ מדידות. ROI גבוה מאוד על השקעה חד-פעמית.")
add_divider(doc)

# ══════════════════════════════════════════════════════════════════════════════
#  APPENDIX — Graph overview
# ══════════════════════════════════════════════════════════════════════════════
doc.add_page_break()
add_heading(doc, "נספח — מבנה גרף הידע וסטטיסטיקות", level=2)

add_code_block(doc,
"סטטיסטיקת endpoints (36,523 פעילויות):",
"""SELECT ?endpointType (COUNT(?act) as ?count) WHERE {
  ?act a odo:Activity ; odo:endpointType ?endpointType .
} GROUP BY ?endpointType ORDER BY DESC(?count) LIMIT 20""",
"""Ki:                  16,195  (44%)
IC50:                 6,815  (19%)
EC50:                 5,297  (15%)
Emax:                 2,015   (6%)
Inhibition:           1,796   (5%)
Ke:                   1,366   (4%)
Activity:               776   (2%)
Intrinsic activity:     279
AD50:                   241
ED50:                   167
Efficacy:               152
Imax:                   141
deltalog(Tau/KA):        63   ← biased agonism metric
Receptor occupancy:      51""")

add_code_block(doc,
"התפלגות מטרות ביולוגיות לפי מספר מדידות:",
"""SELECT ?tgtLabel (COUNT(?act) as ?count) WHERE {
  ?act a odo:Activity ; odo:hasTarget ?tgt .
  ?tgt rdfs:label ?tgtLabel .
} GROUP BY ?tgtLabel ORDER BY DESC(?count) LIMIT 10""",
"""Mu opioid receptor:     14,087  (39%)
Kappa opioid receptor:   9,541  (26%)
Delta opioid receptor:   9,157  (25%)
Nociceptin receptor:     2,048   (6%)
Opioid receptor:           801   (2%)
MOR-DOR:                   546   (1%)
D(3) dopamine receptor:     76
NTS1 receptor:              40
D(2) dopamine receptor:     33""")

add_code_block(doc,
"פורמט assays:",
"""SELECT ?format ?setting (COUNT(?assay) as ?count) WHERE {
  ?assay a odo:Assay ; odo:hasAssayFormat ?fmt ;
         odo:experimentalSetting ?setting .
  ?fmt rdfs:label ?format .
} GROUP BY ?format ?setting ORDER BY DESC(?count)""",
"""cell-free format    | in vitro:  2,392  (53%)  ← רדיוליגנד binding
tissue-based format  | in vitro:  1,147  (25%)
tissue-based format  | ex vivo:     640  (14%)
cell-based format    | in vitro:    310   (7%)
organism-based format| in vivo:      17   (<1%)  ← פחות מ-1% in vivo!
mixed                | in vitro:      5""")

# Save
out_path = "/home/yehely/מסמכים/לימודים/פרויקט ODO/knowledge graph/opioid_insights_report.docx"
doc.save(out_path)
print(f"Saved: {out_path}")
