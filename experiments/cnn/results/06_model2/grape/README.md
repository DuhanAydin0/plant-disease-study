==============================
MODEL-2 TRAINING STARTED
Plant: grape
Classes (4): ['Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)', 'healthy']
Device: mps
==============================

Epoch [01/20] Train Loss: 0.9510 | Train Acc: 0.5572 || Val Loss: 0.6280 | Val Acc: 0.7632
Epoch [02/20] Train Loss: 0.4072 | Train Acc: 0.8403 || Val Loss: 0.2582 | Val Acc: 0.9030
Epoch [03/20] Train Loss: 0.2124 | Train Acc: 0.9191 || Val Loss: 0.2507 | Val Acc: 0.9079
Epoch [04/20] Train Loss: 0.1496 | Train Acc: 0.9469 || Val Loss: 0.1629 | Val Acc: 0.9474
Epoch [05/20] Train Loss: 0.1288 | Train Acc: 0.9539 || Val Loss: 0.1330 | Val Acc: 0.9507
Epoch [06/20] Train Loss: 0.1025 | Train Acc: 0.9592 || Val Loss: 0.1256 | Val Acc: 0.9523
Epoch [07/20] Train Loss: 0.0858 | Train Acc: 0.9708 || Val Loss: 0.1676 | Val Acc: 0.9375
Epoch [08/20] Train Loss: 0.0703 | Train Acc: 0.9712 || Val Loss: 0.1085 | Val Acc: 0.9622
Epoch [09/20] Train Loss: 0.0662 | Train Acc: 0.9750 || Val Loss: 0.1371 | Val Acc: 0.9507
Epoch [10/20] Train Loss: 0.0917 | Train Acc: 0.9712 || Val Loss: 0.1312 | Val Acc: 0.9556
Epoch [11/20] Train Loss: 0.0691 | Train Acc: 0.9761 || Val Loss: 0.1007 | Val Acc: 0.9605
Epoch [12/20] Train Loss: 0.0557 | Train Acc: 0.9792 || Val Loss: 0.1068 | Val Acc: 0.9655
Epoch [13/20] Train Loss: 0.0437 | Train Acc: 0.9866 || Val Loss: 0.0925 | Val Acc: 0.9737
Epoch [14/20] Train Loss: 0.0362 | Train Acc: 0.9884 || Val Loss: 0.1206 | Val Acc: 0.9638
Epoch [15/20] Train Loss: 0.0090 | Train Acc: 0.9972 || Val Loss: 0.0924 | Val Acc: 0.9753
Epoch [16/20] Train Loss: 0.0092 | Train Acc: 0.9972 || Val Loss: 0.1439 | Val Acc: 0.9671
Epoch [17/20] Train Loss: 0.0089 | Train Acc: 0.9965 || Val Loss: 0.1027 | Val Acc: 0.9737
Epoch [18/20] Train Loss: 0.0190 | Train Acc: 0.9930 || Val Loss: 0.1216 | Val Acc: 0.9704

⛔ Early stopping triggered. Best epoch: 15


(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/evaluate_model2.py --plant grape
=== Evaluation Complete ===
{
  "accuracy": 0.967266775777414,
  "macro_recall": 0.9734641339514221,
  "weighted_recall": 0.967266775777414,
  "macro_f1": 0.9719434489616343,
  "num_classes": 4,
  "classes": [
    "Black_rot",
    "Esca_(Black_Measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)",
    "healthy"
  ]
}

Confusion Matrix:
[[170   5   1   1]
 [ 10 198   0   0]
 [  3   0 159   0]
 [  0   0   0  64]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant grape
=== Class-wise Recall ===
Black_rot                     : 0.9605
Esca_(Black_Measles)          : 0.9519
Leaf_blight_(Isariopsis_Leaf_Spot): 0.9815
healthy                       : 1.0000
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant grape
=== Logit Margin Analysis ===
Mean margin: 16.1452
Std margin : 10.2324
Percentiles:
  10%: 4.3614
  25%: 9.5437
  50%: 14.6510
  75%: 21.6857
  90%: 30.0863


🍇 Grape — MODEL-2 Gözlemleri (Multi-class, 4 sınıf)
Grape bitkisi için MODEL-2, tüm çok sınıflı bitkiler arasında en güçlü ve en net ayrışmanın elde edildiği örneklerden biri olmuştur. Eğitim süreci hızlı bir şekilde yakınsamış, validation performansı istikrarlı biçimde artmış ve 15. epoch’ta en iyi validation loss elde edilerek early stopping doğru noktada tetiklenmiştir. Bu noktadan sonra görülen küçük dalgalanmalar, modelin doygunluğa ulaştığını göstermektedir.
Test seti üzerinde %96.7 accuracy ve 0.97 macro recall elde edilmiştir. Macro recall’ın accuracy’den yüksek olması, modelin sınıflar arasında son derece dengeli davrandığını ve hiçbir sınıfı baskın hale getirmeden öğrendiğini göstermektedir. Sınıf bazlı recall değerleri incelendiğinde en düşük performans Esca (0.9519) sınıfında gözlenmiş olsa da, bu değer dahi yüksek kabul edilecek bir seviyededir. Healthy sınıfında recall = 1.00 olup, sağlıklı yaprakların hastalıklardan net biçimde ayrıldığı görülmektedir.
Confusion matrix, hataların sınırlı ve dağınık olduğunu göstermektedir. En fazla karışma Black_rot ↔ Esca arasında gözlenmiş, ancak bu karışma oranı düşük seviyede kalmıştır. Leaf_blight sınıfı ise diğer sınıflardan büyük ölçüde ayrışmıştır. Bu durum, Grape hastalıklarının görsel olarak birbirinden daha belirgin morfolojik özellikler taşıdığını düşündürmektedir.
Margin analizi Grape için MODEL-2’nin en güçlü karar sınırını ortaya koymaktadır. Ortalama logit margin ≈ 16.15 olup, median değerin de yüksek olması modelin kararlarını yüksek güvenle verdiğini göstermektedir. Alt %10’luk dilimde dahi margin değerlerinin 4.36 seviyesinde olması, Corn ve Apple’a kıyasla çok daha az belirsiz örnek bulunduğunu doğrulamaktadır. Bu sonuçlar, Grape dataset’inin MODEL-2 mimarisi için son derece uygun ve iyi ayrışan bir problem sunduğunu göstermektedir.