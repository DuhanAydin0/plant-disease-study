import json
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import importlib.util

# repo_root = .../plant disease study
REPO_ROOT = Path(__file__).resolve().parents[5]

RESULTS_DIR = REPO_ROOT / "experiments" / "cnn" / "results" / "03_all_dataset"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DEF_FILE = REPO_ROOT / "experiments" / "cnn" / "runs" / "03_all_dataset" / "cnn_03_all_model.py"
CNN_CKPT = REPO_ROOT / "experiments" / "cnn" / "results" / "03_all_dataset" / "cnn_03_all_dataset_30epochs_model.pth"
DATA_SPLIT = REPO_ROOT / "data" / "processed" / "full_split"


def load_cnn_class(model_file: Path):
    spec = importlib.util.spec_from_file_location("cnn_03_all_model", str(model_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module.CNN_03_All_Dataset


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)
    print("saving to:", RESULTS_DIR)

    CNN_03_All_Dataset = load_cnn_class(MODEL_DEF_FILE)


    test_dataset_tmp = datasets.ImageFolder(DATA_SPLIT / "test")
    num_classes = len(test_dataset_tmp.classes)

    model = CNN_03_All_Dataset(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(CNN_CKPT, map_location=device))
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(DATA_SPLIT / "test", transform=tfm)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    margins = []
    correct_flags = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            top2 = torch.topk(logits, k=2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()

            preds = torch.argmax(logits, dim=1)
            correct = preds.eq(labels).detach().cpu().numpy()

            margins.extend(margin.tolist())
            correct_flags.extend(correct.tolist())
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    margins = np.array(margins)
    correct_flags = np.array(correct_flags, dtype=bool)

    stats = {
        "margin_definition": "logit top1-top2",
        "mean_margin_all": float(margins.mean()),
        "mean_margin_correct": float(margins[correct_flags].mean()) if correct_flags.any() else None,
        "mean_margin_wrong": float(margins[~correct_flags].mean()) if (~correct_flags).any() else None,
        "median_margin_correct": float(np.median(margins[correct_flags])) if correct_flags.any() else None,
        "median_margin_wrong": float(np.median(margins[~correct_flags])) if (~correct_flags).any() else None,
        "n_total": int(len(margins)),
        "n_correct": int(correct_flags.sum()),
        "n_wrong": int((~correct_flags).sum()),
    }

    # Save
    write_json(RESULTS_DIR / "margin_stats.json", stats)

    lines = ["true_label,pred_label,correct,margin"]
    for yt, yp, ok, m in zip(y_true, y_pred, correct_flags.astype(int), margins):
        lines.append(f"{yt},{yp},{ok},{m:.8f}")
    (RESULTS_DIR / "per_sample_margin.csv").write_text("\n".join(lines), encoding="utf-8")

    print("saved:", RESULTS_DIR / "margin_stats.json")
    print("saved:", RESULTS_DIR / "per_sample_margin.csv")


if __name__ == "__main__":
    main()
