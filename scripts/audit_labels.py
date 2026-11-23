import sys
sys.path.insert(0, 'src')
import csv
import json
from pathlib import Path
import re
from invoice_vision.parse.spatial_rules import validate_gstin_checksum, position_aware_gstin_correct

with open('data/splits_v3.json', encoding='utf-8') as f:
    splits = json.load(f)

val_clean = splits.get('combined_clean_splits', {}).get('val', [])
test_clean = splits.get('combined_clean_splits', {}).get('test', [])

with open('data/ground_truth_gst.csv', newline='', encoding='utf-8') as f:
    public_gt = {r['filename']: r for r in csv.DictReader(f)}

classes = {0: 'vendor', 1: 'invoice_no', 2: 'date', 3: 'gstin', 4: 'table', 5: 'total'}

def audit_split(split_name, img_list):
    print(f"=== AUDIT FOR {split_name.upper()} SPLIT ===")
    results_a = []
    results_b = []
    
    for fn in img_list:
        stem = Path(fn).stem
        lbl_p = Path(f'data/yolo_v2/labels/{split_name}/{stem}.txt')
        counts = {c: 0 for c in classes.values()}
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding='utf-8').splitlines():
                parts = line.split()
                cid = int(parts[0])
                if cid in classes:
                    counts[classes[cid]] += 1
                    
        is_public = fn.startswith('gst_')
        gt_gstins = set()
        gt_has = {'invoice_no': False, 'date': False, 'total': False}
        
        if is_public:
            r = public_gt.get(fn, {})
            sg = r.get('supplier_gst', '').strip()
            cg = r.get('customer_gst', '').strip()
            for cand in [sg, cg]:
                if not cand:
                    continue
                if validate_gstin_checksum(cand):
                    gt_gstins.add(cand)
                else:
                    corr = position_aware_gstin_correct(cand)
                    if validate_gstin_checksum(corr):
                        gt_gstins.add(corr)
            if r.get('invoice_number', '').strip():
                gt_has['invoice_no'] = True
            if r.get('purchase_date', '').strip():
                gt_has['date'] = True
            if r.get('total_amount', '').strip():
                gt_has['total'] = True
        else:
            pdf_p = Path(f'data/processed/pdf_text/{stem}.json')
            if pdf_p.exists():
                d = json.loads(pdf_p.read_text(encoding='utf-8'))
                words = d.get('words', [])
                full_text = ' '.join(w['text'] for w in words)
                for cand in re.findall(r'[0-9A-Za-z]{15}', full_text):
                    if validate_gstin_checksum(cand):
                        gt_gstins.add(cand)
                    else:
                        corr = position_aware_gstin_correct(cand)
                        if validate_gstin_checksum(corr):
                            gt_gstins.add(corr)
                
                # Check whether fields exist in own bills
                if stem in ['own_001_p1', 'own_002_p1', 'own_003_p1', 'own_010_p1', 'own_011_p1', 'own_012_p1', 'own_013_p1']:
                    gt_has['invoice_no'] = True
                    gt_has['date'] = True
                    gt_has['total'] = True
                elif stem == 'own_010_p2':
                    gt_has['invoice_no'] = False
                    gt_has['date'] = False
                    gt_has['total'] = False
                elif stem.startswith('own_02'):
                    gt_has['invoice_no'] = True
                    gt_has['date'] = True
                    gt_has['total'] = True

        gt_gst_cnt = len(gt_gstins)
        human_gst_cnt = counts['gstin']
        
        gt_exceeds_boxes = gt_gst_cnt > human_gst_cnt
        results_a.append({
            'filename': fn,
            'gt_gstin_count': gt_gst_cnt,
            'human_gstin_boxes': human_gst_cnt,
            'gt_exceeds': gt_exceeds_boxes
        })
        
        missing_fields = []
        for fld in ['invoice_no', 'date', 'total']:
            if gt_has[fld] and counts[fld] == 0:
                missing_fields.append(fld)
        if missing_fields:
            results_b.append({
                'filename': fn,
                'missing_classes': missing_fields,
            })
            
    print("--- (a) GSTIN String Count vs Human Boxes ---")
    print("| Filename | GT Valid GSTIN Count | Human GSTIN Boxes | GT > Boxes? |")
    print("| :--- | :---: | :---: | :---: |")
    for ra in results_a:
        print(f"| `{ra['filename']}` | {ra['gt_gstin_count']} | {ra['human_gstin_boxes']} | **{ra['gt_exceeds']}** |")
        
    print("\n--- (b) Non-Empty GT with 0 Human Boxes ---")
    if results_b:
        print("| Filename | Class Name | Human Box Count | GT Status |")
        print("| :--- | :--- | :---: | :--- |")
        for rb in results_b:
            for cls_name in rb['missing_classes']:
                print(f"| `{rb['filename']}` | `{cls_name}` | 0 | Non-empty in Ground Truth |")
    else:
        print("None found.")
    print("\n")

audit_split('val', val_clean)
audit_split('test', test_clean)
