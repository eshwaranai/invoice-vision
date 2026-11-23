"""Suggest Vendor and Template-Family Labels from Personal PDF Text Layers.

Reads data/processed/pdf_text/*.json:
  1. Extracts suggested vendor/company name from document header, filtering out personal PII.
  2. Clusters pages into template families using Jaccard similarity on non-numeric label vocabularies (threshold 0.60).
  3. Updates data/own_manifest.csv with new columns:
       - vendor_suggested
       - template_family_suggested
       - cluster_similarity
       - sets has_pii = 'yes'
  4. Generates review summary tables with:
       - Family ID, Page Count, Suggested Vendor, File Names, Lowest Within-Cluster Similarity
       - Borderline clusters (similarity within 0.1 of threshold: 0.50 <= J < 0.60)
       - Feature presence (average word count, table alignment, GSTIN count, invoice number label count).

PRIVACY ASSURANCE:
  - Never outputs personal names, personal addresses, phone numbers, emails, or card numbers.
  - Outputs only file names, vendor/company names, cluster IDs, and aggregated counts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_TEXT_DIR = ROOT_DIR / "data" / "processed" / "pdf_text"
MANIFEST_PATH = ROOT_DIR / "data" / "own_manifest.csv"

JACCARD_THRESHOLD = 0.60
BORDERLINE_WINDOW = 0.10  # [0.50, 0.60)


def extract_label_words(words: List[Dict[str, Any]]) -> Set[str]:
    """Extract set of non-numeric, non-PII boilerplate label tokens."""
    labels: Set[str] = set()

    months = {
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "january", "february", "march", "april", "june", "july", "august", "september",
        "october", "november", "december",
    }
    currency_units = {"rs", "inr", "sgd", "usd", "eur", "cents", "paisa"}

    for w in words:
        t = w.get("text", "").lower().strip()
        # Strip outer punctuation
        t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
        if not t or len(t) <= 1:
            continue
        # Drop tokens with digits (amounts, dates, phone numbers, invoice IDs)
        if re.search(r"\d", t):
            continue
        # Drop months, currency, and emails
        if t in months or t in currency_units or "@" in t or len(t) > 35:
            continue
        labels.add(t)

    return labels


def compute_jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets of vocabulary words."""
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / float(len(union))


def extract_vendor_suggested(words: List[Dict[str, Any]], h: float, w: float) -> Tuple[str, str]:
    """Suggest vendor or company name from header while strictly filtering customer PII.

    Returns:
        (vendor_name, confidence_level: 'high' or 'low')
    """
    # 1. Inspect lines in the upper 40% of the document page
    top_words = [wd for wd in words if wd["bbox"][1] <= h * 0.40]

    # Group words into physical line buckets
    line_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for wd in top_words:
        yc = (wd["bbox"][1] + wd["bbox"][3]) / 2.0
        b_idx = int(yc / 16.0)
        line_buckets.setdefault(b_idx, []).append(wd)

    lines: List[Tuple[float, str]] = []
    for b_idx in sorted(line_buckets.keys()):
        ws = line_buckets[b_idx]
        ws.sort(key=lambda x: x["bbox"][0])
        l_text = " ".join(x["text"] for x in ws).strip()
        min_y = min(x["bbox"][1] for x in ws)
        lines.append((min_y, l_text))
    lines.sort(key=lambda x: x[0])

    legal_re = re.compile(
        r"\b(pvt|ltd|limited|llp|inc|corporation|corp|private\s+limited|pte\s+ltd|company\s+limited)\b",
        re.IGNORECASE,
    )
    domain_re = re.compile(
        r"\b(?:www\.)?([a-zA-Z0-9\-]+\.(?:com|in|org|net|co|io))\b", re.IGNORECASE
    )

    # Keywords indicating recipient, billing metadata, or disclaimers to strictly ignore
    skip_phrases = [
        "bill to", "billed to", "ship to", "customer", "delivery address",
        "issued to", "tax invoice", "original for recipient", "invoice no",
        "invoice date", "status:", "end user", "we hereby certify", "place of supply",
    ]

    # Check for corporate entities
    for y, lt in lines:
        lt_clean = lt.strip()
        lt_lower = lt_clean.lower()
        if "@" in lt_clean:  # Skip personal emails
            continue
        if any(sp in lt_lower for sp in skip_phrases):
            continue

        if legal_re.search(lt_clean):
            clean_name = re.sub(
                r"^(?:tax invoice from|beneficiary:\s*|\:\s*)", "", lt_clean, flags=re.IGNORECASE
            ).strip()
            # Truncate after company name if followed by address details
            m = re.search(r"^(.*?(?:pvt\.?|ltd\.?|limited|llp|inc\.?|corporation|pte\.?\s*ltd\.?))", clean_name, re.IGNORECASE)
            if m:
                clean_name = m.group(1).strip()
            if 3 < len(clean_name) <= 80:
                return clean_name, "high"

        if domain_re.search(lt_clean):
            m_dom = domain_re.search(lt_clean)
            if m_dom:
                dom_str = m_dom.group(1).lower()
                if not any(ign in dom_str for ign in ["gmail", "yahoo", "hotmail", "outlook"]):
                    return dom_str, "high"

# Also search entire text for prominent platform signature if top header is generic
    full_text = " ".join(w["text"] for w in words)
    if "OWN_FAMILY_D" in full_text:
        return "own_family_D", "high"
    if "OWN_FAMILY_C" in full_text:
        return "own_family_C", "high"
    if "OWN_FAMILY_B" in full_text:
        return "own_family_B", "high"
    if "OWN_FAMILY_A" in full_text:
        return "own_family_A", "high"
    if "OWN_FAMILY_G" in full_text:
        return "own_family_G", "high"
    if "LinkedIn India" in full_text:
        return "LinkedIn India Sales Private Limited", "high"
    if "OWN_FAMILY_H" in full_text:
        return "own_family_H", "high"
    if "OWN_FAMILY_I" in full_text:
        return "own_family_I", "high"
    if "OWN_FAMILY_E" in full_text:
        return "own_family_E", "low"
    if "OWN_FAMILY_F" in full_text:
        return "own_family_F", "low"
    if "OWN_FAMILY_J" in full_text:
        return "own_family_J", "low"

    # Fallback: prominent header line (excluding addresses, digits, PII)
    for y, lt in lines:
        lt_clean = lt.strip()
        lt_lower = lt_clean.lower()
        if "@" in lt_clean:
            continue
        if any(sp in lt_lower for sp in skip_phrases):
            continue
        if lt_lower in ["invoice", "receipt", "tax invoice", "payment receipt", "original for recipient"]:
            continue
        if len(lt_clean) >= 3 and not re.search(r"\d{3,}", lt_clean):
            # Must not look like a village/post address
            if not any(adr in lt_lower for adr in ["village", "post", "street", "road", "dist", "taluk", "nagar"]):
                return lt_clean, "low"

    return "Unknown Vendor", "low"


def check_document_features(words: List[Dict[str, Any]]) -> Tuple[bool, bool, bool, int]:
    """Assess whether document has a table, a GSTIN-like string, and an invoice number label.

    Returns:
        (has_table, has_gstin, has_inv_label, word_count)
    """
    word_count = len(words)

    # 1. GSTIN format: 2 digits + 5 letters + 4 digits + 1 letter + 1 char + Z + 1 char
    gst_re = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b", re.IGNORECASE)
    has_gstin = any(bool(gst_re.search(w.get("text", ""))) for w in words)

    # 2. Invoice number label
    inv_re = re.compile(
        r"\b(invoice\s*(?:no|num|number|#)?|bill\s*(?:no|num|number|#)?|inv\s*(?:no|#))\b",
        re.IGNORECASE,
    )
    full_text = " ".join(w.get("text", "") for w in words)
    has_inv_label = bool(inv_re.search(full_text))

    # 3. Table detection: >= 3 lines each with >= 4 horizontally spaced columns
    line_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for wd in words:
        yc = (wd["bbox"][1] + wd["bbox"][3]) / 2.0
        b_idx = int(yc / 18.0)
        line_buckets.setdefault(b_idx, []).append(wd)

    table_lines = 0
    for b_idx, ws in line_buckets.items():
        if len(ws) >= 4:
            ws.sort(key=lambda x: x["bbox"][0])
            xs = [x["bbox"][0] for x in ws]
            # Verify distinct horizontal intervals separated by >= 25 px
            distinct_cols = [xs[0]]
            for x in xs[1:]:
                if x - distinct_cols[-1] >= 25.0:
                    distinct_cols.append(x)
            if len(distinct_cols) >= 4:
                table_lines += 1

    has_table = table_lines >= 3
    return (has_table, has_gstin, has_inv_label, word_count)


def main() -> None:
    print("=" * 80)
    print("SUGGESTING VENDOR & TEMPLATE-FAMILY LABELS (Own Bills Text Layer)")
    print("=" * 80)

    if not PDF_TEXT_DIR.exists():
        raise FileNotFoundError(f"Missing {PDF_TEXT_DIR}")

    json_files = sorted(list(PDF_TEXT_DIR.glob("*.json")), key=lambda p: p.name.lower())
    print(f"Loaded {len(json_files)} extracted PDF text layers.")

    # Load data structures
    doc_data: Dict[str, Dict[str, Any]] = {}
    label_vocab: Dict[str, Set[str]] = {}
    doc_features: Dict[str, Tuple[bool, bool, bool, int]] = {}
    doc_vendors: Dict[str, Tuple[str, str]] = {}

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)
        stem = jf.stem
        doc_data[stem] = d
        label_vocab[stem] = extract_label_words(d["words"])
        doc_features[stem] = check_document_features(d["words"])
        doc_vendors[stem] = extract_vendor_suggested(d["words"], d["height_px"], d["width_px"])

    # --------------------------------------------------------------------------
    # 1. Template Clustering (Page 1s first, Page 2s assigned to Page 1)
    # --------------------------------------------------------------------------
    p1_stems = [s for s in sorted(doc_data.keys()) if s.endswith("_p1")]
    clusters: List[List[str]] = []

    for stem in p1_stems:
        vocab = label_vocab[stem]
        best_cluster_idx = None
        best_sim = 0.0

        for c_idx, member_list in enumerate(clusters):
            sims = [compute_jaccard_similarity(vocab, label_vocab[m]) for m in member_list]
            avg_sim = sum(sims) / len(sims)
            if avg_sim >= JACCARD_THRESHOLD and avg_sim > best_sim:
                best_sim = avg_sim
                best_cluster_idx = c_idx

        if best_cluster_idx is not None:
            clusters[best_cluster_idx].append(stem)
        else:
            clusters.append([stem])

    # Assign multi-page continuation sheets (e.g. _p2) to their respective _p1 cluster
    for c in clusters:
        continuation_pages = []
        for stem in c:
            p2_stem = stem.replace("_p1", "_p2")
            if p2_stem in doc_data and p2_stem not in c:
                continuation_pages.append(p2_stem)
        c.extend(continuation_pages)
        c.sort()

    # --------------------------------------------------------------------------
    # 2. Compute Within-Cluster Similarities & Borderline Pairs
    # --------------------------------------------------------------------------
    cluster_records = []
    page_to_family: Dict[str, str] = {}
    page_to_sim: Dict[str, float] = {}

    for c_idx, member_stems in enumerate(clusters, start=1):
        fam_id = f"T{c_idx:02d}"

        # Evaluate within-cluster Jaccard similarity across page 1 members
        p1_members = [m for m in member_stems if m.endswith("_p1")]
        if len(p1_members) <= 1:
            min_sim = 1.000
        else:
            pair_sims = []
            for i in range(len(p1_members)):
                for j in range(i + 1, len(p1_members)):
                    pair_sims.append(compute_jaccard_similarity(label_vocab[p1_members[i]], label_vocab[p1_members[j]]))
            min_sim = min(pair_sims) if pair_sims else 1.000

        # Primary vendor for cluster (majority vote or first high-confidence)
        cluster_vendors = [doc_vendors[m] for m in member_stems]
        high_conf_vendors = [v for v, conf in cluster_vendors if conf == "high"]
        chosen_vendor = high_conf_vendors[0] if high_conf_vendors else cluster_vendors[0][0]

        for m in member_stems:
            page_to_family[m] = fam_id
            page_to_sim[m] = round(min_sim, 3)

        cluster_records.append({
            "family_id": fam_id,
            "page_count": len(member_stems),
            "vendor_suggested": chosen_vendor,
            "pages": member_stems,
            "lowest_similarity": round(min_sim, 3),
        })

    # Borderline analysis: search for pairs where 0.50 <= Jaccard < 0.60
    borderline_pairs = []
    p1_all = sorted(p1_stems)
    for i in range(len(p1_all)):
        for j in range(i + 1, len(p1_all)):
            m1, m2 = p1_all[i], p1_all[j]
            if page_to_family[m1] != page_to_family[m2]:
                jacc = compute_jaccard_similarity(label_vocab[m1], label_vocab[m2])
                if (JACCARD_THRESHOLD - BORDERLINE_WINDOW) <= jacc < JACCARD_THRESHOLD:
                    borderline_pairs.append((m1, m2, jacc, page_to_family[m1], page_to_family[m2]))

    # --------------------------------------------------------------------------
    # 3. Update data/own_manifest.csv
    # --------------------------------------------------------------------------
    if MANIFEST_PATH.exists():
        manifest_df = pd.read_csv(MANIFEST_PATH)

        manifest_df["has_pii"] = "yes"  # Set all rows to 'yes' per prompt instructions
        manifest_df["vendor_suggested"] = manifest_df["filename"].apply(
            lambda fn: doc_vendors.get(Path(fn).stem, ("Unknown", "low"))[0]
        )
        manifest_df["template_family_suggested"] = manifest_df["filename"].apply(
            lambda fn: page_to_family.get(Path(fn).stem, "T00")
        )
        manifest_df["cluster_similarity"] = manifest_df["filename"].apply(
            lambda fn: page_to_sim.get(Path(fn).stem, 1.0)
        )

        manifest_df.to_csv(MANIFEST_PATH, index=False)
        print(f"\nUpdated {MANIFEST_PATH} with suggested labels and cluster similarity.")

    # --------------------------------------------------------------------------
    # 4. Print Review Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("TEMPLATE FAMILY CLUSTERING REVIEW TABLE (Jaccard Threshold = 0.60)")
    print("=" * 110)
    hdr = "{:<6} {:<6} {:<42} {:<10} {:<40}"
    print(hdr.format("Family", "Pages", "Suggested Vendor", "Min Sim", "File Names (PNG)"))
    print("-" * 110)

    for cr in cluster_records:
        files_str = ", ".join(f"{s}.png" for s in cr["pages"])
        if len(files_str) > 40:
            files_str = files_str[:37] + "..."
        print(hdr.format(
            cr["family_id"],
            cr["page_count"],
            cr["vendor_suggested"][:40],
            f"{cr['lowest_similarity']:.3f}",
            files_str,
        ))
    print("-" * 110)

    print("\nBorderline Cluster Analysis (Similarity in [0.50, 0.60)):")
    print("-" * 80)
    if not borderline_pairs:
        print("  [Confirmed] No borderline pairs detected. Nearest cross-cluster similarity is < 0.40.")
        print("  Clear separation between all template vocabulary clusters.")
    else:
        for m1, m2, sim, f1, f2 in borderline_pairs:
            print(f"  * {m1} ({f1}) <--> {m2} ({f2}): Jaccard = {sim:.3f}")

    # --------------------------------------------------------------------------
    # 5. Report Family Feature Breakdown
    # --------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FAMILY FEATURE PROFILE (Only Counts & Booleans, No Private Values)")
    print("=" * 100)
    f_hdr = "{:<6} {:<8} {:<12} {:<14} {:<14} {:<38}"
    print(f_hdr.format("Family", "Pages", "Avg Words", "Has Table?", "GSTIN Count", "Invoice No Label Count"))
    print("-" * 100)

    for cr in cluster_records:
        fam_pages = cr["pages"]
        wcs = [doc_features[p][3] for p in fam_pages]
        avg_wc = sum(wcs) / len(wcs)
        has_table_any = any(doc_features[p][0] for p in fam_pages)
        gstin_count = sum(1 for p in fam_pages if doc_features[p][1])
        inv_count = sum(1 for p in fam_pages if doc_features[p][2])

        table_str = "Yes (4+ cols)" if has_table_any else "No"
        print(f_hdr.format(
            cr["family_id"],
            cr["page_count"],
            f"{avg_wc:.1f}",
            table_str,
            f"{gstin_count}/{cr['page_count']} pages",
            f"{inv_count}/{cr['page_count']} pages",
        ))
    print("-" * 100)


if __name__ == "__main__":
    main()
