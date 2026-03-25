
==============================
MODEL-2 TRAINING STARTED
Plant: peach
Classes (2): ['Bacterial_spot', 'healthy']
Device: mps
==============================

Epoch [01/30] Train Loss: 0.4478 | Train Acc: 0.8509 || Val Loss: 0.4036 | Val Acc: 0.8643 | LR: 1.00e-05
Epoch [02/30] Train Loss: 0.3997 | Train Acc: 0.8649 || Val Loss: 0.3824 | Val Acc: 0.8643 | LR: 1.00e-05
Epoch [03/30] Train Loss: 0.3641 | Train Acc: 0.8649 || Val Loss: 0.3459 | Val Acc: 0.8643 | LR: 1.00e-05
Epoch [04/30] Train Loss: 0.3178 | Train Acc: 0.8649 || Val Loss: 0.2861 | Val Acc: 0.8643 | LR: 1.00e-05
Epoch [05/30] Train Loss: 0.2571 | Train Acc: 0.8649 || Val Loss: 0.2286 | Val Acc: 0.8643 | LR: 1.00e-05
Epoch [06/30] Train Loss: 0.2079 | Train Acc: 0.8767 || Val Loss: 0.1939 | Val Acc: 0.8719 | LR: 1.00e-05
Epoch [07/30] Train Loss: 0.1777 | Train Acc: 0.9042 || Val Loss: 0.1777 | Val Acc: 0.8769 | LR: 1.00e-05
Epoch [08/30] Train Loss: 0.1609 | Train Acc: 0.9273 || Val Loss: 0.1534 | Val Acc: 0.9221 | LR: 1.00e-05
Epoch [09/30] Train Loss: 0.1458 | Train Acc: 0.9456 || Val Loss: 0.1404 | Val Acc: 0.9673 | LR: 1.00e-05
Epoch [10/30] Train Loss: 0.1349 | Train Acc: 0.9607 || Val Loss: 0.1313 | Val Acc: 0.9698 | LR: 1.00e-05
Epoch [11/30] Train Loss: 0.1285 | Train Acc: 0.9639 || Val Loss: 0.1197 | Val Acc: 0.9573 | LR: 1.00e-05
Epoch [12/30] Train Loss: 0.1175 | Train Acc: 0.9672 || Val Loss: 0.1116 | Val Acc: 0.9724 | LR: 1.00e-05
Epoch [13/30] Train Loss: 0.1087 | Train Acc: 0.9682 || Val Loss: 0.1073 | Val Acc: 0.9548 | LR: 1.00e-05
Epoch [14/30] Train Loss: 0.1027 | Train Acc: 0.9726 || Val Loss: 0.0947 | Val Acc: 0.9749 | LR: 1.00e-05
Epoch [15/30] Train Loss: 0.0958 | Train Acc: 0.9763 || Val Loss: 0.0880 | Val Acc: 0.9749 | LR: 1.00e-05
Epoch [16/30] Train Loss: 0.0869 | Train Acc: 0.9801 || Val Loss: 0.0845 | Val Acc: 0.9648 | LR: 1.00e-05
Epoch [17/30] Train Loss: 0.0843 | Train Acc: 0.9758 || Val Loss: 0.0742 | Val Acc: 0.9824 | LR: 1.00e-05
Epoch [18/30] Train Loss: 0.0785 | Train Acc: 0.9769 || Val Loss: 0.0695 | Val Acc: 0.9849 | LR: 1.00e-05
Epoch [19/30] Train Loss: 0.0726 | Train Acc: 0.9801 || Val Loss: 0.0707 | Val Acc: 0.9774 | LR: 1.00e-05
Epoch [20/30] Train Loss: 0.0739 | Train Acc: 0.9806 || Val Loss: 0.0622 | Val Acc: 0.9824 | LR: 1.00e-05
Epoch [21/30] Train Loss: 0.0721 | Train Acc: 0.9812 || Val Loss: 0.0578 | Val Acc: 0.9849 | LR: 1.00e-05
Epoch [22/30] Train Loss: 0.0684 | Train Acc: 0.9790 || Val Loss: 0.0575 | Val Acc: 0.9874 | LR: 1.00e-05
Epoch [23/30] Train Loss: 0.0646 | Train Acc: 0.9860 || Val Loss: 0.0530 | Val Acc: 0.9824 | LR: 1.00e-05
Epoch [24/30] Train Loss: 0.0651 | Train Acc: 0.9790 || Val Loss: 0.0495 | Val Acc: 0.9874 | LR: 1.00e-05
Epoch [25/30] Train Loss: 0.0600 | Train Acc: 0.9849 || Val Loss: 0.0471 | Val Acc: 0.9874 | LR: 1.00e-05
Epoch [26/30] Train Loss: 0.0572 | Train Acc: 0.9849 || Val Loss: 0.0445 | Val Acc: 0.9874 | LR: 1.00e-05
Epoch [27/30] Train Loss: 0.0546 | Train Acc: 0.9833 || Val Loss: 0.0424 | Val Acc: 0.9899 | LR: 1.00e-05
Epoch [28/30] Train Loss: 0.0513 | Train Acc: 0.9849 || Val Loss: 0.0413 | Val Acc: 0.9925 | LR: 1.00e-05
Epoch [29/30] Train Loss: 0.0533 | Train Acc: 0.9871 || Val Loss: 0.0429 | Val Acc: 0.9849 | LR: 1.00e-05
Epoch [30/30] Train Loss: 0.0483 | Train Acc: 0.9871 || Val Loss: 0.0424 | Val Acc: 0.9849 | LR: 1.00e-05
✅ Model saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/peach/best_model.pth
📊 Metrics saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/peach/metrics.json
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/evaluate_model2.py --plant peach
=== Evaluation Complete ===
{
  "accuracy": 0.9900249376558603,
  "macro_recall": 0.9789280084077772,
  "weighted_recall": 0.9900249376558603,
  "macro_f1": 0.9789280084077772,
  "num_classes": 2,
  "classes": [
    "Bacterial_spot",
    "healthy"
  ]
}

Confusion Matrix:
[[344   2]
 [  2  53]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant peach
=== Class-wise Recall ===
Bacterial_spot                : 0.9942
healthy                       : 0.9636
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant peach
=== Logit Margin Analysis ===
Mean margin: 10.3571
Std margin : 6.4236
Percentiles:
  10%: 2.0576
  25%: 4.9450
  50%: 9.6161
  75%: 15.9052
  90%: 19.1634
