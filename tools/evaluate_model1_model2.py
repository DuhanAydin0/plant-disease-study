# tools/evaluate_model1_model2.py

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from inference.backends.model1_model2 import Model1Model2Backend  # noqa: E402
from inference.labels import plant_id_from_model1_label  # noqa: E402


IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Sadece gerçekten mismatch olan disease adları:
DISEASE_ALIASES = {
    # global folder: "Cercospora_leaf_spot Gray_leaf_spot"
    "Cercospora_leaf_spot_Gray_leaf_spot": "Cercospora_leaf_spot Gray_leaf_spot",
    # global folder: "Spider_mites Two-spotted_spider_mite"
    "Spider_mites_Two-spotted_spider_mite": "Spider_mites Two-spotted_spider_mite",
}


def iter_images_flat(test_root: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for class_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        true_label = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((true_label, p))
    return items


def build_plantid_to_rawlabel(model1_class_to_idx_path: Path) -> Dict[str, str]:
    """
    Model1 raw label'ları: 'Pepper,_bell' gibi.
    plant_id: 'pepper_bell' (model2 folder id)
    Bu map ile global label prefix'ini dataset ile aynı tutuyoruz.
    """
    with open(model1_class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    plantid_to_raw: Dict[str, str] = {}
    for raw_label in class_to_idx.keys():
        pid = plant_id_from_model1_label(raw_label)
        plantid_to_raw[pid] = raw_label
    return plantid_to_raw


def canonicalize_global_label(label: str, valid_labels: set) -> str:
    """
    label valid değilse, disease alias uygula.
    """
    if label in valid_labels:
        return label
    if "___" not in label:
        return label

    plant, disease = label.split("___", 1)
    disease2 = DISEASE_ALIASES.get(disease, disease)
    fixed = f"{plant}___{disease2}"
    return fixed if fixed in valid_labels else label


def predict_global_label(pred: Dict, plantid_to_raw: Dict[str, str], valid_labels: set) -> str:
    """
    Backend output -> global label (folder name ile eşleşecek).
    """
    plant_id = pred.get("plant_id")
    raw_plant = plantid_to_raw.get(plant_id)

    mode = pred.get("mode")
    if mode == "singleclass_ood":
        if pred.get("ood", False):
            return "UNKNOWN"
        disease = pred.get("disease")
        out = f"{raw_plant}___{disease}"
        return canonicalize_global_label(out, valid_labels)

    # multiclass_or_binary
    disease = pred.get("disease")
    out = f"{raw_plant}___{disease}"
    return canonicalize_global_label(out, valid_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True, help="Path to full_split/test")
    ap.add_argument("--out", required=True, help="Output json path (e.g., reports/model1_model2_eval.json)")
    ap.add_argument(
        "--model1-class-to-idx",
        default=str(ROOT / "experiments" / "cnn" / "results" / "05_model1" / "model1_class_to_idx.json"),
        help="Path to model1_class_to_idx.json",
    )
    args = ap.parse_args()

    test_root = Path(args.test_root)
    out_path = Path(args.out)
    model1_class_to_idx_path = Path(args.model1_class_to_idx)

    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")
    if not model1_class_to_idx_path.exists():
        raise FileNotFoundError(f"model1_class_to_idx.json not found: {model1_class_to_idx_path}")

    # Valid label set folder names (+ UNKNOWN)
    true_labels = sorted([p.name for p in test_root.iterdir() if p.is_dir()])
    valid_labels = set(true_labels) | {"UNKNOWN"}

    print("\n==============================")
    print("EVALUATE: MODEL1 + MODEL2 PIPELINE (FIXED LABELS)")
    print(f"Test root: {test_root}")
    print(f"Num labels (folders): {len(true_labels)}")
    print("==============================\n")

    plantid_to_raw = build_plantid_to_rawlabel(model1_class_to_idx_path)
    backend = Model1Model2Backend()

    items = iter_images_flat(test_root)
    if not items:
        raise RuntimeError(f"No images found under: {test_root}")

    y_true: List[str] = []
    y_pred: List[str] = []
    mismatches: List[Dict] = []
    ood_count = 0

    for true_label, img_path in items:
        pred = backend.predict_one(str(img_path))
        pred_label = predict_global_label(pred, plantid_to_raw, valid_labels)

        # true label da canonicalize 
        true_fixed = canonicalize_global_label(true_label, valid_labels)

        y_true.append(true_fixed)
        y_pred.append(pred_label)

        if pred.get("mode") == "singleclass_ood" and pred.get("ood", False):
            ood_count += 1

        if pred_label != true_fixed and len(mismatches) < 40:
            mismatches.append(
                {
                    "image": str(img_path),
                    "true": true_fixed,
                    "pred": pred_label,
                    "raw": pred,
                }
            )

    labels_all = sorted(set(y_true) | set(y_pred))
    report_dict = classification_report(y_true, y_pred, labels=labels_all, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=labels_all, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_all).tolist()

    out = {
        "meta": {
            "test_root": str(test_root),
            "num_samples": len(y_true),
            "num_labels": len(labels_all),
            "ood_pred_count": int(ood_count),
            "disease_aliases": DISEASE_ALIASES,
        },
        "summary": {
            "accuracy": float(report_dict.get("accuracy", 0.0)),
            "macro_recall": float(report_dict.get("macro avg", {}).get("recall", 0.0)),
            "weighted_recall": float(report_dict.get("weighted avg", {}).get("recall", 0.0)),
            "macro_f1": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
        },
        "classification_report": report_dict,
        "labels": labels_all,
        "confusion_matrix": cm,
        "sample_mismatches": mismatches,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report_text)
    print("\n===== SUMMARY =====")
    print(json.dumps(out["summary"], indent=2))
    print(f"OOD preds: {ood_count}")
    print(f"Saved -> {out_path}\n")

    # En düşük recall'lar , bunu global cnn modelimin düşük recall değerleri ile karşılaştırmak için kullanıyorum.
    per_class = []
    for cls in labels_all:
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        if cls in report_dict:
            per_class.append((cls, report_dict[cls].get("recall", 0.0), report_dict[cls].get("support", 0)))
    per_class.sort(key=lambda x: x[1])

    print("===== LOWEST 15 CLASS RECALLS =====")
    for cls, rec, sup in per_class[:15]:
        print(f"{rec:7.4f} | sup={int(sup):4d} | {cls}")


if __name__ == "__main__":
    main()
