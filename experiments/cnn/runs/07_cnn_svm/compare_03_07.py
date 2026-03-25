from pathlib import Path
import json
import csv
import pandas as pd


# -------------------------------------------------
# PATHS (GitHub indexed)
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]

CNN03_DIR = REPO_ROOT / "experiments" / "cnn" / "results" / "03_all_dataset"
SVM07_DIR = REPO_ROOT / "experiments" / "cnn" / "results" / "07_cnn_svm"

OUT_MD = SVM07_DIR / "REPORT_03_vs_07.md"


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt4(x):
    if x is None:
        return "NA"
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def df_to_md_table(df: pd.DataFrame) -> str:
    """No tabulate dependency."""
    if df is None or df.empty:
        return "_No data_\n"

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    return "\n".join([header, sep] + rows) + "\n"


# -------------------------------------------------
# Robust read for per_class_recall.csv
# Supports:
#  - class,recall,support
#  - class,precision,recall,f1,support
# and survives extra commas.
# -------------------------------------------------
def read_per_class_recall_csv(path: Path, recall_col_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    header = [h.strip() for h in rows[0]]
    header_l = [h.lower() for h in header]

    # Header-based indices if possible
    i_class = header_l.index("class") if "class" in header_l else None
    i_recall = header_l.index("recall") if "recall" in header_l else None
    i_support = header_l.index("support") if "support" in header_l else None

    parsed = []

    if i_class is not None and i_recall is not None:
        for r in rows[1:]:
            if not r:
                continue
            # pad row to header length
            while len(r) < len(header):
                r.append("")
            cls = r[i_class].strip()
            if not cls:
                continue
            try:
                rec = float(r[i_recall])
            except ValueError:
                continue
            sup = None
            if i_support is not None:
                try:
                    sup = int(float(r[i_support]))
                except ValueError:
                    sup = None
            parsed.append((cls, rec, sup))

    # Fallback: last two tokens are recall/support; rest is class
    if not parsed:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            cls = ",".join(parts[:-2]).strip()
            rec_str = parts[-2].strip()
            sup_str = parts[-1].strip()
            try:
                rec = float(rec_str)
                sup = int(float(sup_str))
            except ValueError:
                continue
            parsed.append((cls, rec, sup))

    if not parsed:
        raise ValueError(f"Could not parse any valid rows from: {path}")

    df = pd.DataFrame(parsed, columns=["class", recall_col_name, "support"])
    # If support entirely missing, drop
    if df["support"].isna().all():
        df = df.drop(columns=["support"])
    else:
        # fill NaN supports with -1 for stable printing
        df["support"] = df["support"].fillna(-1).astype(int)

    return df


# -------------------------------------------------
# Margin summary from per_sample_margin.csv
# Works with:
# - true_label,pred_label,correct,margin
# - true_label,pred_label,correct,margin,true_name,pred_name
# -------------------------------------------------
def compute_margin_stats_from_per_sample_csv(path: Path):
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None

    header = [h.strip() for h in lines[0].split(",")]
    # locate indices (fallback 2,3)
    try:
        i_correct = header.index("correct")
    except ValueError:
        i_correct = 2
    try:
        i_margin = header.index("margin")
    except ValueError:
        i_margin = 3

    correct = []
    wrong = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) <= max(i_correct, i_margin):
            continue
        try:
            ok = int(float(parts[i_correct]))
            m = float(parts[i_margin])
        except ValueError:
            continue
        if ok == 1:
            correct.append(m)
        else:
            wrong.append(m)

    import numpy as np
    mc = np.array(correct, dtype=float)
    mw = np.array(wrong, dtype=float)

    return {
        "n_correct": int(mc.size),
        "n_wrong": int(mw.size),
        "mean_margin_correct": float(mc.mean()) if mc.size else None,
        "mean_margin_wrong": float(mw.mean()) if mw.size else None,
        "median_margin_correct": float(np.median(mc)) if mc.size else None,
        "median_margin_wrong": float(np.median(mw)) if mw.size else None,
    }


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    # Your real filenames
    cnn03_recall = CNN03_DIR / "per_class_recall.csv"
    svm07_recall = SVM07_DIR / "per_class_recall.csv"

    if not cnn03_recall.exists():
        raise FileNotFoundError(f"Missing Global CNN recall CSV:\n{cnn03_recall}")
    if not svm07_recall.exists():
        raise FileNotFoundError(f"Missing CNN+SVM recall CSV:\n{svm07_recall}")

    df03 = read_per_class_recall_csv(cnn03_recall, "recall_cnn03")
    df07 = read_per_class_recall_csv(svm07_recall, "recall_svm07")

    merged = df03.merge(df07[["class", "recall_svm07"]], on="class", how="inner")
    merged["delta_recall"] = merged["recall_svm07"] - merged["recall_cnn03"]

    # Worst/best sections
    worst_cnn03 = merged.sort_values("recall_cnn03", ascending=True).head(15)
    best_gain = merged.sort_values("delta_recall", ascending=False).head(15)
    worst_drop = merged.sort_values("delta_recall", ascending=True).head(15)

    # Overall summaries
    s03 = load_json_if_exists(CNN03_DIR / "eval_summary.json") or {}
    s07 = load_json_if_exists(SVM07_DIR / "eval_summary.json") or {}

    # Margin stats (json + csv summary)
    m03_json = load_json_if_exists(CNN03_DIR / "margin_stats.json")
    m07_json = load_json_if_exists(SVM07_DIR / "margin_stats.json")

    m03_csv = compute_margin_stats_from_per_sample_csv(CNN03_DIR / "per_sample_margin.csv")
    m07_csv = compute_margin_stats_from_per_sample_csv(SVM07_DIR / "per_sample_margin.csv")

    # Build report
    SVM07_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Global CNN (03) vs CNN+SVM (07)\n")

    lines.append("## Run Paths\n")
    lines.append(f"- CNN03: `{CNN03_DIR}`\n")
    lines.append(f"- SVM07: `{SVM07_DIR}`\n")

    # Overall metrics
    if s03 or s07:
        lines.append("## Overall Metrics (eval_summary.json)\n")
        lines.append("| Metric | CNN03 | SVM07 |")
        lines.append("|---|---:|---:|")
        for k in ["test_accuracy", "macro_recall", "weighted_recall", "macro_f1", "weighted_f1"]:
            lines.append(f"| {k} | {fmt4(s03.get(k))} | {fmt4(s07.get(k))} |")
        lines.append("")

    # Recall comparisons
    lines.append("## Lowest Recall Classes in CNN03 (top-15)\n")
    lines.append(df_to_md_table(worst_cnn03))

    lines.append("## Biggest Recall Gains (SVM07 - CNN03) (top-15)\n")
    lines.append(df_to_md_table(best_gain))

    lines.append("## Biggest Recall Drops (SVM07 - CNN03) (top-15)\n")
    lines.append(df_to_md_table(worst_drop))

    # Margin comparisons
    lines.append("## Margin Comparison\n")
    lines.append(
        "Important: CNN03 margin = **logit top1-top2**; SVM07 margin = **decision-score top1-top2**. "
        "Scales differ; compare separation (correct vs wrong), not absolute values.\n"
    )

    def margin_row(d, name):
        if not d:
            return f"| {name} | NA | NA | NA | NA | NA | NA |"
        return (
            f"| {name} | {fmt4(d.get('mean_margin_correct'))} | {fmt4(d.get('mean_margin_wrong'))} | "
            f"{fmt4(d.get('median_margin_correct'))} | {fmt4(d.get('median_margin_wrong'))} | "
            f"{d.get('n_correct','NA')} | {d.get('n_wrong','NA')} |"
        )

    lines.append("### margin_stats.json\n")
    lines.append("| Source | mean_correct | mean_wrong | median_correct | median_wrong | n_correct | n_wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(margin_row(m03_json, "CNN03 (json)"))
    lines.append(margin_row(m07_json, "SVM07 (json)"))
    lines.append("")

    lines.append("### per_sample_margin.csv (computed)\n")
    lines.append("| Source | mean_correct | mean_wrong | median_correct | median_wrong | n_correct | n_wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(margin_row(m03_csv, "CNN03 (csv)"))
    lines.append(margin_row(m07_csv, "SVM07 (csv)"))
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] saved:", OUT_MD)


if __name__ == "__main__":
    main()
