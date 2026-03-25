==============================
MODEL-2 TRAINING STARTED
Plant: cherry
Classes (2): ['Powdery_mildew', 'healthy']
Device: mps
==============================

Epoch [01/30] Train Loss: 0.6734 | Train Acc: 0.6152 || Val Loss: 0.6484 | Val Acc: 0.9684 | LR: 1.00e-05
Epoch [02/30] Train Loss: 0.6163 | Train Acc: 0.8710 || Val Loss: 0.5689 | Val Acc: 0.8982 | LR: 1.00e-05
Epoch [03/30] Train Loss: 0.5177 | Train Acc: 0.9370 || Val Loss: 0.4528 | Val Acc: 0.9614 | LR: 1.00e-05
Epoch [04/30] Train Loss: 0.3968 | Train Acc: 0.9662 || Val Loss: 0.3341 | Val Acc: 0.9649 | LR: 1.00e-05
Epoch [05/30] Train Loss: 0.2874 | Train Acc: 0.9677 || Val Loss: 0.2435 | Val Acc: 0.9789 | LR: 1.00e-05
Epoch [06/30] Train Loss: 0.2117 | Train Acc: 0.9737 || Val Loss: 0.1795 | Val Acc: 0.9754 | LR: 1.00e-05
Epoch [07/30] Train Loss: 0.1592 | Train Acc: 0.9820 || Val Loss: 0.1447 | Val Acc: 0.9789 | LR: 1.00e-05
Epoch [08/30] Train Loss: 0.1234 | Train Acc: 0.9835 || Val Loss: 0.1176 | Val Acc: 0.9789 | LR: 1.00e-05
Epoch [09/30] Train Loss: 0.1025 | Train Acc: 0.9872 || Val Loss: 0.1037 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [10/30] Train Loss: 0.0850 | Train Acc: 0.9880 || Val Loss: 0.0993 | Val Acc: 0.9825 | LR: 1.00e-05
Epoch [11/30] Train Loss: 0.0748 | Train Acc: 0.9857 || Val Loss: 0.0872 | Val Acc: 0.9719 | LR: 1.00e-05
Epoch [12/30] Train Loss: 0.0698 | Train Acc: 0.9872 || Val Loss: 0.0803 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [13/30] Train Loss: 0.0614 | Train Acc: 0.9902 || Val Loss: 0.0769 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [14/30] Train Loss: 0.0556 | Train Acc: 0.9887 || Val Loss: 0.0733 | Val Acc: 0.9825 | LR: 1.00e-05
Epoch [15/30] Train Loss: 0.0518 | Train Acc: 0.9880 || Val Loss: 0.0688 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [16/30] Train Loss: 0.0469 | Train Acc: 0.9910 || Val Loss: 0.0678 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [17/30] Train Loss: 0.0461 | Train Acc: 0.9917 || Val Loss: 0.0634 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [18/30] Train Loss: 0.0431 | Train Acc: 0.9925 || Val Loss: 0.0616 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [19/30] Train Loss: 0.0387 | Train Acc: 0.9917 || Val Loss: 0.0593 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [20/30] Train Loss: 0.0406 | Train Acc: 0.9932 || Val Loss: 0.0583 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [21/30] Train Loss: 0.0341 | Train Acc: 0.9947 || Val Loss: 0.0571 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [22/30] Train Loss: 0.0343 | Train Acc: 0.9932 || Val Loss: 0.0554 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [23/30] Train Loss: 0.0352 | Train Acc: 0.9955 || Val Loss: 0.0575 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [24/30] Train Loss: 0.0316 | Train Acc: 0.9940 || Val Loss: 0.0557 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [25/30] Train Loss: 0.0310 | Train Acc: 0.9940 || Val Loss: 0.0565 | Val Acc: 0.9860 | LR: 1.00e-05
Epoch [26/30] Train Loss: 0.0316 | Train Acc: 0.9940 || Val Loss: 0.0529 | Val Acc: 0.9860 | LR: 5.00e-06
Epoch [27/30] Train Loss: 0.0301 | Train Acc: 0.9947 || Val Loss: 0.0513 | Val Acc: 0.9895 | LR: 5.00e-06
Epoch [28/30] Train Loss: 0.0303 | Train Acc: 0.9940 || Val Loss: 0.0510 | Val Acc: 0.9895 | LR: 5.00e-06
Epoch [29/30] Train Loss: 0.0304 | Train Acc: 0.9955 || Val Loss: 0.0507 | Val Acc: 0.9860 | LR: 5.00e-06
Epoch [30/30] Train Loss: 0.0258 | Train Acc: 0.9947 || Val Loss: 0.0505 | Val Acc: 0.9860 | LR: 5.00e-06
✅ Model saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/cherry/best_model.pth
📊 Metrics saved to: /Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/cherry/metrics.json
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/evaluate_model2.py --plant cherry     
=== Evaluation Complete ===
{
  "accuracy": 0.9930555555555556,
  "macro_recall": 0.9937106918238994,
  "weighted_recall": 0.9930555555555556,
  "macro_f1": 0.9929892891918208,
  "num_classes": 2,
  "classes": [
    "Powdery_mildew",
    "healthy"
  ]
}

Confusion Matrix:
[[157   2]
 [  0 129]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant cherry     
=== Class-wise Recall ===
Powdery_mildew                : 0.9874
healthy                       : 1.0000
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant cherry     
=== Logit Margin Analysis ===
Mean margin: 5.8525
Std margin : 1.8321
Percentiles:
  10%: 3.6963
  25%: 4.8237
  50%: 5.9709
  75%: 7.2126
  90%: 7.9774
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % 
