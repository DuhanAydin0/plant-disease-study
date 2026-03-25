==============================
MODEL-2 TRAINING STARTED
Plant: strawberry
Classes (2): ['Leaf_scorch', 'healthy']
Device: mps
==============================

Epoch [01/30] Train Loss: 0.6072 | Train Acc: 0.7068 || Val Loss: 0.5835 | Val Acc: 0.7094 | LR: 1.00e-05
Epoch [02/30] Train Loss: 0.5804 | Train Acc: 0.7087 || Val Loss: 0.5572 | Val Acc: 0.7094 | LR: 1.00e-05
Epoch [03/30] Train Loss: 0.5440 | Train Acc: 0.7087 || Val Loss: 0.5113 | Val Acc: 0.7094 | LR: 1.00e-05
Epoch [04/30] Train Loss: 0.4934 | Train Acc: 0.7087 || Val Loss: 0.4460 | Val Acc: 0.7094 | LR: 1.00e-05
Epoch [05/30] Train Loss: 0.4250 | Train Acc: 0.7333 || Val Loss: 0.3769 | Val Acc: 0.8120 | LR: 1.00e-05
Epoch [06/30] Train Loss: 0.3649 | Train Acc: 0.8073 || Val Loss: 0.3243 | Val Acc: 0.8675 | LR: 1.00e-05
Epoch [07/30] Train Loss: 0.3158 | Train Acc: 0.8822 || Val Loss: 0.2750 | Val Acc: 0.9444 | LR: 1.00e-05
Epoch [08/30] Train Loss: 0.2718 | Train Acc: 0.9315 || Val Loss: 0.2316 | Val Acc: 0.9444 | LR: 1.00e-05
Epoch [09/30] Train Loss: 0.2356 | Train Acc: 0.9333 || Val Loss: 0.2034 | Val Acc: 0.9573 | LR: 1.00e-05
Epoch [10/30] Train Loss: 0.2174 | Train Acc: 0.9333 || Val Loss: 0.1782 | Val Acc: 0.9530 | LR: 1.00e-05
Epoch [11/30] Train Loss: 0.1941 | Train Acc: 0.9507 || Val Loss: 0.1568 | Val Acc: 0.9701 | LR: 1.00e-05
Epoch [12/30] Train Loss: 0.1766 | Train Acc: 0.9416 || Val Loss: 0.1463 | Val Acc: 0.9573 | LR: 1.00e-05
Epoch [13/30] Train Loss: 0.1802 | Train Acc: 0.9361 || Val Loss: 0.1393 | Val Acc: 0.9658 | LR: 1.00e-05
Epoch [14/30] Train Loss: 0.1592 | Train Acc: 0.9479 || Val Loss: 0.1260 | Val Acc: 0.9615 | LR: 1.00e-05
Epoch [15/30] Train Loss: 0.1486 | Train Acc: 0.9534 || Val Loss: 0.1208 | Val Acc: 0.9744 | LR: 1.00e-05
Epoch [16/30] Train Loss: 0.1402 | Train Acc: 0.9534 || Val Loss: 0.1085 | Val Acc: 0.9744 | LR: 1.00e-05
Epoch [17/30] Train Loss: 0.1346 | Train Acc: 0.9534 || Val Loss: 0.1038 | Val Acc: 0.9786 | LR: 1.00e-05
Epoch [18/30] Train Loss: 0.1380 | Train Acc: 0.9543 || Val Loss: 0.1093 | Val Acc: 0.9701 | LR: 1.00e-05
Epoch [19/30] Train Loss: 0.1301 | Train Acc: 0.9607 || Val Loss: 0.0949 | Val Acc: 0.9786 | LR: 1.00e-05
Epoch [20/30] Train Loss: 0.1283 | Train Acc: 0.9571 || Val Loss: 0.0928 | Val Acc: 0.9786 | LR: 1.00e-05
Epoch [21/30] Train Loss: 0.1136 | Train Acc: 0.9653 || Val Loss: 0.0892 | Val Acc: 0.9744 | LR: 1.00e-05
Epoch [22/30] Train Loss: 0.1209 | Train Acc: 0.9644 || Val Loss: 0.0849 | Val Acc: 0.9786 | LR: 1.00e-05
Epoch [23/30] Train Loss: 0.1132 | Train Acc: 0.9580 || Val Loss: 0.0823 | Val Acc: 0.9744 | LR: 1.00e-05
Epoch [24/30] Train Loss: 0.1044 | Train Acc: 0.9644 || Val Loss: 0.0780 | Val Acc: 0.9786 | LR: 1.00e-05
Epoch [25/30] Train Loss: 0.1041 | Train Acc: 0.9616 || Val Loss: 0.0763 | Val Acc: 0.9786 | LR: 1.00e-05
Epoch [26/30] Train Loss: 0.1058 | Train Acc: 0.9616 || Val Loss: 0.0724 | Val Acc: 0.9829 | LR: 1.00e-05
Epoch [27/30] Train Loss: 0.0980 | Train Acc: 0.9671 || Val Loss: 0.0703 | Val Acc: 0.9829 | LR: 1.00e-05
Epoch [28/30] Train Loss: 0.0955 | Train Acc: 0.9644 || Val Loss: 0.0685 | Val Acc: 0.9786 | LR: 1.00e-05
Epoch [29/30] Train Loss: 0.0920 | Train Acc: 0.9680 || Val Loss: 0.0658 | Val Acc: 0.9829 | LR: 1.00e-05
Epoch [30/30] Train Loss: 0.0898 | Train Acc: 0.9662 || Val Loss: 0.0659 | Val Acc: 0.9744 | LR: 1.00e-05
✅ Model saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/strawberry/best_model.pth
📊 Metrics saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/strawberry/metrics.json
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/evaluate_model2.py --plant strawberry
=== Evaluation Complete ===
{
  "accuracy": 0.9745762711864406,
  "macro_recall": 0.9607741039659811,
  "weighted_recall": 0.9745762711864406,
  "macro_f1": 0.968736200653537,
  "num_classes": 2,
  "classes": [
    "Leaf_scorch",
    "healthy"
  ]
}

Confusion Matrix:
[[166   1]
 [  5  64]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant strawberry
=== Class-wise Recall ===
Leaf_scorch                   : 0.9940
healthy                       : 0.9275
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant strawberry
=== Logit Margin Analysis ===
Mean margin: 8.3306
Std margin : 5.6406
Percentiles:
  10%: 1.8535
  25%: 2.9411
  50%: 7.7305
  75%: 12.1957
  90%: 16.3112
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % 
