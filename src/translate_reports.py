"""Clean + translate PTB-XL free-text reports (German / Swedish / English) to English with MarianMT on CPU."""
import os, re, sys, time
import pandas as pd, torch
from transformers import MarianMTModel, MarianTokenizer

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(6)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTB = os.environ.get('PTB_DIR', os.path.join(REPO_ROOT, 'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'))
OUT = os.environ.get('REPORTS_EN_CSV', os.path.join(REPO_ROOT, 'data', 'ptbxl_reports_en.csv'))

df = pd.read_csv(os.path.join(PTB, 'ptbxl_database.csv'), index_col='ecg_id')


def clean(s):
    s = str(s) if isinstance(s, str) else ''
    s = s.lower()                                   # fixes "vÄnster" -> "vänster"
    s = re.sub(r'\d\.\d\d\s+unbest[äa]tigter bericht', ' ', s)   # "4.46 unconfirmed report" boilerplate
    s = re.sub(r'unbest[äa]tigter bericht', ' ', s)
    s = re.sub(r'edit:.*$', ' ', s)                 # annotator edit notes
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# German cardiology glossary applied BEFORE machine translation (MarianMT mistranslates this jargon,
# e.g. "überdrehter linkstyp" -> "overturned left type" instead of "left axis deviation").
U = '(?:ü|ue)'
A = '(?:ä|ae)'
O = '(?:ö|oe)'
GLOSSARY = [
    # --- multi-word / most specific first ---
    (r'\balt(?:er)? unbest(?:\.|immt)?', 'age undetermined.'),
    (r'wahrscheinlich alt\b', 'probably old.'),
    (r'verdacht auf', 'suspected'),
    (r'p-sinistro(?:cardiale|atriale)|p sinistro(?:cardiale|atriale)', 'left atrial enlargement.'),
    (r'p-dextro(?:cardiale|atriale)|p dextro(?:cardiale|atriale)', 'right atrial enlargement.'),
    (r'p-verbreiterung', 'p wave widening.'),
    (rf'qt-verl{A}ngerung', 'prolonged qt interval.'),
    (rf'm{A}(?:ss|ß)ige amplitudenkriterien f(?:ü|u|#|ue)r', 'moderate voltage criteria for'),
    (r'amplitudenkriterien f(?:ü|u|#|ue)r', 'voltage criteria for'),
    (rf'isch{A}mie oder linksbelastung', 'ischemia or left ventricular strain.'),
    (rf'isch{A}mie', 'ischemia'),
    (r'innenschichts?schaden', 'subendocardial injury.'),
    (r'myokardschaden', 'myocardial injury.'),
    (r'nicht auszuschlie(?:ß|ss)en', 'cannot be excluded.'),
    (r'st & t abnorm(?:al)?', 'st-t wave abnormality.'),
    (r'qrs\(t\) abnorm(?:al)?', 'qrs(t) abnormal.'),
    (r'schwierig bestimmbare qrs-achse', 'indeterminate qrs axis.'),
    (rf'{U}berdrehter linkstyp|extrem(?:er)? linkstyp', 'left axis deviation.'),
    (rf'{U}berdrehter rechtstyp|extrem(?:er)? rechtstyp', 'right axis deviation.'),
    (r'linkslagetyp|linkstyp', 'left axis.'),
    (r'rechtslagetyp|rechtstyp', 'right axis.'),
    (r'lagetyp normal|normaltyp|indifferenztyp', 'normal axis.'),
    (r'steiltyp', 'vertical axis.'),
    (r'sagittaltyp', 'sagittal axis.'),
    (r'tachykardes vorhofflimmern', 'atrial fibrillation with rapid ventricular response.'),
    (r'vorhofflimmern/-flattern|vorhofflimmern', 'atrial fibrillation.'),
    (r'vorhofflattern', 'atrial flutter.'),
    (r'sinusrhythmus', 'sinus rhythm.'),
    (r'sinusbradykardie', 'sinus bradycardia.'),
    (r'sinustachykardie', 'sinus tachycardia.'),
    (r'sinusarrhythmie|sinus arrhythmie', 'sinus arrhythmia.'),
    (rf'supraventrikul{A}re arrhythmie', 'supraventricular arrhythmia.'),
    (r'sonst normales ekg', 'otherwise normal ecg.'),
    (r'normales ekg', 'normal ecg.'),
    (r'grenzwertiges ekg', 'borderline ecg.'),
    (r'pathologisches ekg', 'abnormal ecg.'),
    (r'periphere niederspannung', 'low voltage in limb leads.'),
    (r'niederspannung', 'low voltage.'),
    (rf'(?:inkompletter|unvollst{A}ndiger) rechtsschenkelblock|(?:inkomplett(?:er)?|unvollst{A}ndiger) rsb', 'incomplete right bundle branch block.'),
    (rf'(?:inkompletter|unvollst{A}ndiger) linksschenkelblock|(?:inkomplett(?:er)?|unvollst{A}ndiger) lsb', 'incomplete left bundle branch block.'),
    (rf'unvollst{A}ndiger', 'incomplete'),
    (r'kompletter rechtsschenkelblock|rechtsschenkelblock', 'right bundle branch block.'),
    (r'kompletter linksschenkelblock|linksschenkelblock', 'left bundle branch block.'),
    (r'part\. rsb|partieller rsb', 'incomplete right bundle branch block.'),
    (r'\brsb\b', 'right bundle branch block.'),
    (r'\blsb\b', 'left bundle branch block.'),
    (r'linksanteriorer hemiblock', 'left anterior fascicular block.'),
    (r'linksposteriorer hemiblock', 'left posterior fascicular block.'),
    (rf'bifaszikul{A}rer block', 'bifascicular block.'),
    (r'av-block 1\.? grades|av block i\.? grades|a-v block i\b|av-block i\b', 'first degree av block.'),
    (rf'unspezifischer intraventrikul{A}rer block', 'non-specific intraventricular block.'),
    (rf'unspezifische intraventrikul{A}re leitungsst{O}rung', 'non-specific intraventricular conduction disturbance.'),
    (rf'intraventrikul{A}re leitungsst{O}rung', 'intraventricular conduction disturbance.'),
    (r'\bintraventr\b\.?', 'intraventricular'),
    (rf'supraventr\. extrasystole(?:n|\(n\))?|supraventrikul{A}re extrasystole\(n\)|supraventrikul{A}re extrasystolen?', 'supraventricular premature complexes.'),
    (rf'ventrikul{A}re extrasystole\(n\)|ventrikul{A}re extrasystolen?', 'ventricular premature complexes.'),
    (r'vorhofs?-es\b', 'atrial premature complexes.'),
    (r'bigeminie', 'bigeminy.'),
    (r'schrittmacher', 'pacemaker.'),
    (rf'linksherzhypertrophie|linksventrikul{A}re hypertrophie|linkshypertrophie', 'left ventricular hypertrophy.'),
    (rf'rechtsherzhypertrophie|rechtsventrikul{A}re hypertrophie|rechtshypertrophie', 'right ventricular hypertrophy.'),
    (r'anteroseptaler infarkt', 'anteroseptal myocardial infarction.'),
    (r'vorderwandinfarkt', 'anterior myocardial infarction.'),
    (r'seitenwandinfarkt|lateralinfarkt', 'lateral myocardial infarction.'),
    (r'hinterwandinfarkt', 'inferior myocardial infarction.'),
    # German adjective endings on locations -> English (inflected forms only)
    (r'\bhochlateral(?:en|er|e|es)\b', 'high lateral'),
    (r'\banterolateral(?:en|er|e|es)\b', 'anterolateral'),
    (r'\binferolateral(?:en|er|e|es)\b', 'inferolateral'),
    (r'\banteroseptal(?:en|er|e|es)\b', 'anteroseptal'),
    (r'\banterior(?:en|er|e|es)\b', 'anterior'),
    (r'\binferior(?:en|er|e|es)\b', 'inferior'),
    (r'\blateral(?:en|er|e|es)\b', 'lateral'),
    (r'\bseptal(?:en|er|e|es)\b', 'septal'),
    (r'\binfarkt\b', 'myocardial infarction.'),
    (r'subakuter?', 'subacute'), (r'\bakuter?\b', 'acute'), (r'\balter\b', 'old'),
    (r'wahrscheinlich', 'probable'), (rf'm{O}glicherweise|m{O}glich|\bm{O}gl\b\.?', 'possible'),
    (r'unspezifische abnorme st-senkung|unspezifische st-senkung', 'non-specific st depression.'),
    (r'st-senkung|st senkung', 'st depression'),
    (r'st-hebung|st hebung', 'st elevation'),
    (rf't-ver{A}nderung(?:en)?', 't wave changes'),
    (r'unspezifisches abnorm(?:es|ales) t|unspezifische t-abflachung|\bt abnorm(?:al)?', 'non-specific t wave abnormality.'),
    (r'schlechte r-progression|fehlende r-progr?ession', 'poor r wave progression.'),
    (r'\bextrem\b', 'extreme'),
    (r'ableitungen', 'leads'), (rf'\b{U}ber\b', 'in'), (r'pathologisch', 'abnormal.'),
    (r'\bekg\b', 'ecg'), (r'\b\d\.\d\d\b', ' '), (r'\boder\b', 'or'), (r'\bund\b', 'and'),
]
GLOSSARY = [(re.compile(p), r) for p, r in GLOSSARY]


def pre_translate(s):
    for p, r in GLOSSARY:
        s = p.sub(r, s)
    return re.sub(r'\s+', ' ', s).strip()


EN = re.compile(r'\b(sinus rhythm|the|normal ecg|waves|segments|consistent|rhythm\.|probable|possible|otherwise)\b')
SV = re.compile(r'(sinusrytm|förmak|vänster|höger|kammar|myokardaffektion|sannolik|avvikande|ospecifik)')


def lang(s):
    if SV.search(s):
        return 'sv'
    if EN.search(s):
        return 'en'
    return 'de'


df['report_clean'] = df['report'].apply(clean)
df['lang'] = df['report_clean'].apply(lang)
print(df['lang'].value_counts().to_dict(), flush=True)

KNOWN = set(r.strip(' .') for _, r in GLOSSARY)


def segments(s):
    return [x.strip() for x in s.split('.') if x.strip()]


def mt(model_name, texts, tag):
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name).eval()
    out_map, t0 = {}, time.time()
    texts = sorted(texts, key=len)
    for i in range(0, len(texts), 48):
        b = texts[i:i + 48]
        enc = tok(b, return_tensors='pt', padding=True, truncation=True, max_length=200)
        with torch.no_grad():
            out = mdl.generate(**enc, num_beams=1, max_new_tokens=160)
        for src, o in zip(b, tok.batch_decode(out, skip_special_tokens=True)):
            out_map[src] = o.strip().rstrip('.').lower()
        if (i // 48) % 25 == 0:
            print(f'{tag} {i}/{len(texts)} {time.time()-t0:.0f}s | {b[-1][:60]} -> {out_map[b[-1]][:80]}', flush=True)
    return out_map


# German: glossary first, then MT only on the phrases the glossary did not cover
de_pre = {s: pre_translate(s) for s in set(df.loc[df['lang'] == 'de', 'report_clean'])}
todo = sorted({seg for p in de_pre.values() for seg in segments(p) if seg not in KNOWN})
print('de unique reports', len(de_pre), '| phrases needing MT', len(todo), flush=True)
de_mt = mt('Helsinki-NLP/opus-mt-de-en', todo, 'de')
de_final = {s: '. '.join(seg if seg in KNOWN else de_mt.get(seg, seg) for seg in segments(p)) + '.'
            for s, p in de_pre.items()}

sv_src = sorted(set(df.loc[df['lang'] == 'sv', 'report_clean']) - {''})
sv_pre = {x: re.sub(r'el-axel', 'axel', x) for x in sv_src}  # MarianMT: 'el-axel' -> 'electric shaft'
print('sv unique', len(sv_src), flush=True)
sv_out = mt('Helsinki-NLP/opus-mt-sv-en', sorted(set(sv_pre.values())), 'sv')
sv_mt = {x: sv_out[sv_pre[x]] for x in sv_src}


def final(s, l):
    if not s:
        return ''
    if l == 'de':
        return de_final[s]
    if l == 'sv':
        return sv_mt.get(s, s) + '.'
    return s


df['report_en'] = [final(s, l) for s, l in zip(df['report_clean'], df['lang'])]
df[['report', 'report_clean', 'lang', 'report_en']].to_csv(OUT, encoding='utf-8')
print('TRANSLATE_DONE', OUT, flush=True)
for s in df.sample(12, random_state=3)[['report_clean', 'report_en']].itertuples():
    print(' *', s.report_clean[:110], '\n   ->', s.report_en[:140])
