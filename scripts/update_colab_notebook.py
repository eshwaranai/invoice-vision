"""Update notebooks/colab_train.ipynb with Experiment 3 cells (yolo_v3)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NB_PATH = ROOT_DIR / "notebooks" / "colab_train.ipynb"


def update_notebook() -> None:
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check if Experiment 3 cells already exist
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "## 11. Experiment 3" in src or "v3_s0" in src:
            print("Experiment 3 cells already present in colab_train.ipynb.")
            return

    c11_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Experiment 3: Extract Expanded Multi-Layout Synthetic Dataset (`yolo_v3`)\n",
            "\n",
            "Extracts `invoice-vision/yolo_v3.zip` containing 117 train images (33 real + 84 synthetic multi-layout), 12 val, and 14 test images. Verifies counts and split integrity prior to training.",
        ],
    }

    c11_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            'YOLO_V3_ZIP = Path("/content/drive/MyDrive/invoice-vision/yolo_v3.zip")\n',
            "assert YOLO_V3_ZIP.exists(), f\"Missing {YOLO_V3_ZIP}. Please upload yolo_v3.zip to Drive under 'invoice-vision/'.\"\n",
            "\n",
            "# Unpack dataset into /content/invoice-vision/data/yolo_v3\n",
            "!mkdir -p /content/invoice-vision/data/yolo_v3\n",
            "!unzip -q -o /content/drive/MyDrive/invoice-vision/yolo_v3.zip -d /content/invoice-vision/data/yolo_v3\n",
            "\n",
            'v3_train = len(list(Path("/content/invoice-vision/data/yolo_v3/images/train").glob("*")))\n',
            'v3_val = len(list(Path("/content/invoice-vision/data/yolo_v3/images/val").glob("*")))\n',
            'v3_test = len(list(Path("/content/invoice-vision/data/yolo_v3/images/test").glob("*")))\n',
            "\n",
            'print("yolo_v3 dataset extracted successfully:")\n',
            'print(f"  Train : {v3_train} images (expected 117: 33 real + 84 synthetic multi-layout)")\n',
            'print(f"  Val   : {v3_val} images (expected 12, identical to yolo_v2)")\n',
            'print(f"  Test  : {v3_test} images (expected 14, identical to yolo_v2)")\n',
            "\n",
            'assert v3_train == 117, f"Expected 117 train images, found {v3_train}"\n',
            'assert v3_val == 12, f"Expected 12 val images, found {v3_val}"\n',
            'assert v3_test == 14, f"Expected 14 test images, found {v3_test}"',
        ],
    }

    c12_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 12. Experiment 3: Multi-Seed Training (Seeds 0 – 4, VAL Evaluation Only)\n",
            "\n",
            "Trains YOLOv8n across 5 independent seeds (`seed=0, 1, 2, 3, 4`) on `data/yolo_v3` with run names `v3_s0` to `v3_s4`.\n",
            "After each run:\n",
            "- Evaluates strictly on **VAL** split (`--splits val`).\n",
            "- Synchronizes model weights, validation predictions, and reports to Google Drive under `invoice-vision/runs/<run-name>`.",
        ],
    }

    c12_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "\n",
            "exp3_seeds = [0, 1, 2, 3, 4]\n",
            "\n",
            "for seed in exp3_seeds:\n",
            '    run_name = f"v3_s{seed}"\n',
            '    print(f"\\n{\'=\'*70}")\n',
            '    print(f"STARTING EXPERIMENT 3 RUN: {run_name} (Seed: {seed})")\n',
            '    print(f"{\'=\'*70}")\n',
            "    \n",
            "    # 1. Train detector on yolo_v3\n",
            "    !python scripts/train_yolo.py \\\n",
            "        --config configs/yolo_v1.yaml \\\n",
            "        --data-root data/yolo_v3 \\\n",
            "        --seed {seed} \\\n",
            "        --run-name {run_name}\n",
            "    \n",
            "    # 2. Locate weights (supporting nested Ultralytics path)\n",
            '    cand_nested = Path(f"runs/detect/runs/detect/{run_name}/weights/best.pt")\n',
            '    cand_flat = Path(f"runs/detect/{run_name}/weights/best.pt")\n',
            "    best_weights = str(cand_nested if cand_nested.exists() else cand_flat)\n",
            "    run_dir = cand_nested.parent.parent if cand_nested.exists() else cand_flat.parent.parent\n",
            "    \n",
            "    # 3. Evaluate strictly on VAL split\n",
            '    val_out_md = f"results/val_eval_{run_name}.md"\n',
            "    \n",
            "    !python scripts/eval_yolo.py \\\n",
            "        --weights {best_weights} \\\n",
            "        --data-root data/yolo_v3 \\\n",
            "        --splits val \\\n",
            "        --imgsz 1024 \\\n",
            "        --out {val_out_md} \\\n",
            "        --overlays-dir results/overlays\n",
            "    \n",
            "    # 4. Synchronize run outputs and report to Google Drive\n",
            '    drive_dest = f"/content/drive/MyDrive/invoice-vision/runs/{run_name}"\n',
            "    !mkdir -p {drive_dest}\n",
            "    !cp -r {run_dir}/* {drive_dest}/\n",
            "    !cp {val_out_md} {drive_dest}/\n",
            '    print(f"Completed {run_name}. Artifacts backed up to {drive_dest}.")\n',
        ],
    }

    c13_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13. Experiment 3: Validation Evaluation & Pre-Registered Decision Rule (v3 vs. Layouts Baseline)\n",
            "\n",
            "Evaluates validation performance against the pre-registered decision rule:\n",
            "1. **Mean Gain Requirement**: $\\Delta \\text{mAP50} \\ge +0.05$ over the layouts baseline ($0.446 \\to \\ge 0.496$).\n",
            "2. **Consistency Requirement**: At least 4 of 5 same-seed pairs favor `data/yolo_v3` over `data/yolo_v2`.\n",
            '3. If both pass: confirm empirical improvement. Otherwise: report "no evidence of improvement".',
        ],
    }

    c13_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import re\n",
            "import pandas as pd\n",
            "from pathlib import Path\n",
            "\n",
            "# Measured validation mAP50 for seeds 0 - 4 from Experiment 2 (docs/experiments.md)\n",
            "# Layouts (data/yolo_v2, 57 images) baseline values:\n",
            "layouts_map50 = {\n",
            "    0: 0.514,\n",
            "    1: 0.460,\n",
            "    2: 0.408,\n",
            "    3: 0.416,\n",
            "    4: 0.430,\n",
            "}\n",
            "layouts_mean_map50 = 0.446\n",
            "\n",
            "v3_map50 = {}\n",
            'classes = ["vendor", "invoice_no", "date", "gstin", "table", "total"]\n',
            "\n",
            "for seed in [0, 1, 2, 3, 4]:\n",
            '    run_name = f"v3_s{seed}"\n',
            '    report_path = Path(f"results/val_eval_{run_name}.md")\n',
            "    if not report_path.exists():\n",
            '        print(f"Notice: {report_path} not found. Skipping.")\n',
            "        continue\n",
            "    \n",
            '    txt = report_path.read_text(encoding="utf-8")\n',
            "    for line in txt.splitlines():\n",
            '        if "Overall mAP@0.50" in line or "| **all** |" in line or "mAP50" in line:\n',
            '            m = re.search(r"(\\d+\\.\\d+)", line)\n',
            "            if m:\n",
            "                v3_map50[seed] = float(m.group(1))\n",
            "                break\n",
            "\n",
            'print("\\n" + "="*75)\n',
            'print("EXPERIMENT 3: VALIDATION DECISION RULE EVALUATION (v3 vs. Layouts)")\n',
            'print("="*75)\n',
            "if v3_map50:\n",
            "    v3_mean = sum(v3_map50.values()) / len(v3_map50)\n",
            "    gain = v3_mean - layouts_mean_map50\n",
            "    \n",
            "    pairs_favoring = sum(1 for s in v3_map50 if s in layouts_map50 and v3_map50[s] > layouts_map50[s])\n",
            "    total_pairs = sum(1 for s in v3_map50 if s in layouts_map50)\n",
            "    \n",
            '    print(f"Layouts Baseline Mean mAP50 (seeds 0-4) : {layouts_mean_map50:.4f}")\n',
            '    print(f"yolo_v3 Mean mAP50 ({len(v3_map50)} seeds)              : {v3_mean:.4f} (gain: {gain:+.4f})")\n',
            '    print(f"Same-seed pairs favoring yolo_v3        : {pairs_favoring}/{total_pairs}")\n',
            '    print("-" * 75)\n',
            "    \n",
            "    gain_pass = gain >= 0.05\n",
            "    pairs_pass = pairs_favoring >= 4 if total_pairs >= 4 else (pairs_favoring == total_pairs)\n",
            "    \n",
            "    if gain_pass and pairs_pass:\n",
            '        print("[DECISION RESULT]: EMPIRICAL IMPROVEMENT CONFIRMED (Gain >= +0.05 and >= 4/5 seed pairs).")\n',
            '        print("Decision: Adopt data/yolo_v3 as the new production training baseline.")\n',
            "    else:\n",
            '        print("[DECISION RESULT]: NO EVIDENCE OF IMPROVEMENT.")\n',
            '        print("Decision: Retain data/yolo_v2 (layouts) baseline per pre-registered decision rule.")\n',
            "else:\n",
            '    print("No v3 validation runs found yet.")\n',
            'print("="*75)\n',
        ],
    }

    # Update section header for the final test run from 11 to 14
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            src = "".join(c.get("source", []))
            if "## 11. RUN ONCE" in src:
                c["source"][0] = "## 14. RUN ONCE, after the config is frozen (Test Set Evaluation)\n"

    # Insert before final test cells (which are the last 2 cells)
    nb["cells"].insert(-2, c11_md)
    nb["cells"].insert(-2, c11_code)
    nb["cells"].insert(-2, c12_md)
    nb["cells"].insert(-2, c12_code)
    nb["cells"].insert(-2, c13_md)
    nb["cells"].insert(-2, c13_code)

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Updated {NB_PATH.relative_to(ROOT_DIR)} successfully. Total cells now: {len(nb['cells'])}.")


if __name__ == "__main__":
    update_notebook()
