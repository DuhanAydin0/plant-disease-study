==============================
MODEL-2 TRAINING STARTED
Plant: pepper_bell
Classes (2): ['Bacterial_spot', 'healthy']
Device: mps
==============================

Epoch [01/30] Train Loss: 0.6693 | Train Acc: 0.5968 || Val Loss: 0.6650 | Val Acc: 0.5973 | LR: 1.00e-05
Epoch [02/30] Train Loss: 0.6531 | Train Acc: 0.5979 || Val Loss: 0.6383 | Val Acc: 0.5973 | LR: 1.00e-05
Epoch [03/30] Train Loss: 0.6198 | Train Acc: 0.6176 || Val Loss: 0.5980 | Val Acc: 0.6108 | LR: 1.00e-05
Epoch [04/30] Train Loss: 0.5655 | Train Acc: 0.7435 || Val Loss: 0.5256 | Val Acc: 0.8000 | LR: 1.00e-05
Epoch [05/30] Train Loss: 0.4956 | Train Acc: 0.8169 || Val Loss: 0.4619 | Val Acc: 0.8622 | LR: 1.00e-05
Epoch [06/30] Train Loss: 0.4310 | Train Acc: 0.8538 || Val Loss: 0.3931 | Val Acc: 0.8676 | LR: 1.00e-05
Epoch [07/30] Train Loss: 0.3948 | Train Acc: 0.8596 || Val Loss: 0.3538 | Val Acc: 0.8784 | LR: 1.00e-05
Epoch [08/30] Train Loss: 0.3666 | Train Acc: 0.8567 || Val Loss: 0.3281 | Val Acc: 0.8784 | LR: 1.00e-05
Epoch [09/30] Train Loss: 0.3427 | Train Acc: 0.8741 || Val Loss: 0.3137 | Val Acc: 0.8757 | LR: 1.00e-05
Epoch [10/30] Train Loss: 0.3307 | Train Acc: 0.8723 || Val Loss: 0.3036 | Val Acc: 0.8730 | LR: 1.00e-05
Epoch [11/30] Train Loss: 0.3150 | Train Acc: 0.8879 || Val Loss: 0.2786 | Val Acc: 0.8892 | LR: 1.00e-05
Epoch [12/30] Train Loss: 0.3055 | Train Acc: 0.8758 || Val Loss: 0.2688 | Val Acc: 0.8865 | LR: 1.00e-05
Epoch [13/30] Train Loss: 0.2901 | Train Acc: 0.8885 || Val Loss: 0.2603 | Val Acc: 0.9027 | LR: 1.00e-05
Epoch [14/30] Train Loss: 0.2794 | Train Acc: 0.8937 || Val Loss: 0.2485 | Val Acc: 0.8946 | LR: 1.00e-05
Epoch [15/30] Train Loss: 0.2782 | Train Acc: 0.8989 || Val Loss: 0.2418 | Val Acc: 0.8892 | LR: 1.00e-05
Epoch [16/30] Train Loss: 0.2693 | Train Acc: 0.8954 || Val Loss: 0.2464 | Val Acc: 0.8892 | LR: 1.00e-05
Epoch [17/30] Train Loss: 0.2544 | Train Acc: 0.9012 || Val Loss: 0.2290 | Val Acc: 0.9054 | LR: 1.00e-05
Epoch [18/30] Train Loss: 0.2535 | Train Acc: 0.9041 || Val Loss: 0.2503 | Val Acc: 0.8919 | LR: 1.00e-05
Epoch [19/30] Train Loss: 0.2526 | Train Acc: 0.9041 || Val Loss: 0.2220 | Val Acc: 0.8865 | LR: 1.00e-05
Epoch [20/30] Train Loss: 0.2420 | Train Acc: 0.9041 || Val Loss: 0.2154 | Val Acc: 0.9054 | LR: 1.00e-05
Epoch [21/30] Train Loss: 0.2432 | Train Acc: 0.9099 || Val Loss: 0.2138 | Val Acc: 0.8919 | LR: 1.00e-05
Epoch [22/30] Train Loss: 0.2314 | Train Acc: 0.9099 || Val Loss: 0.2080 | Val Acc: 0.8973 | LR: 1.00e-05
Epoch [23/30] Train Loss: 0.2304 | Train Acc: 0.9162 || Val Loss: 0.2153 | Val Acc: 0.9135 | LR: 1.00e-05
Epoch [24/30] Train Loss: 0.2265 | Train Acc: 0.9151 || Val Loss: 0.2179 | Val Acc: 0.9108 | LR: 1.00e-05
Epoch [25/30] Train Loss: 0.2271 | Train Acc: 0.9185 || Val Loss: 0.2054 | Val Acc: 0.9162 | LR: 1.00e-05
Epoch [26/30] Train Loss: 0.2144 | Train Acc: 0.9237 || Val Loss: 0.2049 | Val Acc: 0.9162 | LR: 1.00e-05
Epoch [27/30] Train Loss: 0.2112 | Train Acc: 0.9168 || Val Loss: 0.1916 | Val Acc: 0.9135 | LR: 1.00e-05
Epoch [28/30] Train Loss: 0.2156 | Train Acc: 0.9226 || Val Loss: 0.2098 | Val Acc: 0.9081 | LR: 1.00e-05
Epoch [29/30] Train Loss: 0.2132 | Train Acc: 0.9243 || Val Loss: 0.1893 | Val Acc: 0.9162 | LR: 1.00e-05
Epoch [30/30] Train Loss: 0.2048 | Train Acc: 0.9237 || Val Loss: 0.1898 | Val Acc: 0.9270 | LR: 1.00e-05
✅ Model saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/pepper_bell/best_model.pth
📊 Metrics saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/pepper_bell/metrics.json
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/evaluate_model2.py --plant pepper_bell
=== Evaluation Complete ===
{
  "accuracy": 0.8983957219251337,
  "macro_recall": 0.8902087726071333,
  "weighted_recall": 0.8983957219251337,
  "macro_f1": 0.8935239293955468,
  "num_classes": 2,
  "classes": [
    "Bacterial_spot",
    "healthy"
  ]
}

Confusion Matrix:
[[128  23]
 [ 15 208]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant pepper_bell
=== Class-wise Recall ===
Bacterial_spot                : 0.8477
healthy                       : 0.9327
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant pepper_bell
=== Logit Margin Analysis ===
Mean margin: 2.5834
Std margin : 1.8385
Percentiles:
  10%: 0.0190
  25%: 1.4392
  50%: 2.8627
  75%: 3.6626
  90%: 4.8889
