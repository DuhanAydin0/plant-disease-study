
==============================
MODEL-2 TRAINING STARTED
Plant: apple
Classes (4): ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'healthy']
Device: mps
==============================

Epoch [01/20] Train Loss: 0.7929 | Train Acc: 0.6785 || Val Loss: 0.3993 | Val Acc: 0.8418
Epoch [02/20] Train Loss: 0.3349 | Train Acc: 0.8819 || Val Loss: 0.2726 | Val Acc: 0.9135
Epoch [03/20] Train Loss: 0.2307 | Train Acc: 0.9179 || Val Loss: 0.2097 | Val Acc: 0.9283
Epoch [04/20] Train Loss: 0.1902 | Train Acc: 0.9355 || Val Loss: 0.1820 | Val Acc: 0.9451
Epoch [05/20] Train Loss: 0.1807 | Train Acc: 0.9369 || Val Loss: 0.2256 | Val Acc: 0.9135
Epoch [06/20] Train Loss: 0.1260 | Train Acc: 0.9554 || Val Loss: 0.1612 | Val Acc: 0.9473
Epoch [07/20] Train Loss: 0.1272 | Train Acc: 0.9585 || Val Loss: 0.1509 | Val Acc: 0.9451
Epoch [08/20] Train Loss: 0.0826 | Train Acc: 0.9693 || Val Loss: 0.1849 | Val Acc: 0.9451
Epoch [09/20] Train Loss: 0.0772 | Train Acc: 0.9716 || Val Loss: 0.1909 | Val Acc: 0.9346
Epoch [10/20] Train Loss: 0.0868 | Train Acc: 0.9707 || Val Loss: 0.1327 | Val Acc: 0.9578
Epoch [11/20] Train Loss: 0.0605 | Train Acc: 0.9775 || Val Loss: 0.1593 | Val Acc: 0.9557
Epoch [12/20] Train Loss: 0.0464 | Train Acc: 0.9842 || Val Loss: 0.1910 | Val Acc: 0.9536
Epoch [13/20] Train Loss: 0.0743 | Train Acc: 0.9725 || Val Loss: 0.2084 | Val Acc: 0.9241

⛔ Early stopping triggered. Best epoch: 10



(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/evaluate_model2.py --plant apple
=== Evaluation Complete ===
{
  "accuracy": 0.9624217118997912,
  "macro_recall": 0.9605913913988815,
  "weighted_recall": 0.9624217118997912,
  "macro_f1": 0.9598697312243041,
  "num_classes": 4,
  "classes": [
    "Apple_scab",
    "Black_rot",
    "Cedar_apple_rust",
    "healthy"
  ]
}sen 

Confusion Matrix:
[[ 88   2   1   4]
 [  2  91   0   1]
 [  0   0  41   1]
 [  6   1   0 241]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant apple
=== Class-wise Recall ===
Apple_scab                    : 0.9263
Black_rot                     : 0.9681
Cedar_apple_rust              : 0.9762
healthy                       : 0.9718
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant apple
=== Logit Margin Analysis ===
Mean margin: 7.1168
Std margin : 4.7213
Percentiles:
  10%: 1.4759
  25%: 3.8106
  50%: 6.5393
  75%: 10.0598
  90%: 14.1421




🍎 Apple — MODEL-2 Gözlemleri (Multi-class, 4 sınıf)
Apple bitkisi için MODEL-2, beklenen şekilde yüksek ve dengeli performans sergilemiştir. Eğitim süreci erken doygunluğa ulaşmış, 10. epoch’ta en düşük validation loss elde edilerek early stopping doğru noktada tetiklenmiştir. Test seti üzerinde elde edilen %96.24 accuracy ve 0.96 macro recall, modelin yalnızca baskın sınıfları değil tüm hastalık sınıflarını dengeli biçimde öğrendiğini göstermektedir.
Sınıf bazlı incelemede, en düşük recall değeri Apple_scab (0.9263) sınıfında gözlenmiştir. Confusion matrix incelendiğinde bu hataların büyük ölçüde Apple_scab ↔ healthy arasında gerçekleştiği görülmektedir. Bu durum, scab hastalığının erken veya hafif evrelerinde görsel belirtilerin sağlıklı yapraklarla örtüşmesi nedeniyle beklenen ve açıklanabilir bir karışmadır. Diğer sınıflarda (Black_rot, Cedar_apple_rust) recall değerleri %96–97 bandındadır.
Margin analizi Apple için güçlü bir karar sınırına işaret etmektedir. Ortalama logit margin ≈ 7.12 olup, median değerin de yüksek olması modelin çoğu örnekte yüksek güvenle karar verdiğini göstermektedir. Düşük margin değerleri yalnızca alt %10’luk dilimde görülmüş ve bu örneklerin confusion matrix’te gözlenen sınır vakalarla örtüştüğü anlaşılmıştır. Genel olarak Apple, MODEL-2 mimarisinin başarılı ve stabil çalıştığı bir referans bitki olarak değerlendirilebilir.