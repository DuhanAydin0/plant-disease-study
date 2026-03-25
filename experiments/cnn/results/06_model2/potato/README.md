==============================
MODEL-2 TRAINING STARTED
Plant: potato
Classes (3): ['Early_blight', 'Late_blight', 'healthy']
Device: mps
==============================

Epoch [01/20] Train Loss: 0.6510 | Train Acc: 0.7178 || Val Loss: 0.3665 | Val Acc: 0.8509
Epoch [02/20] Train Loss: 0.3093 | Train Acc: 0.8851 || Val Loss: 0.1482 | Val Acc: 0.9379
Epoch [03/20] Train Loss: 0.1905 | Train Acc: 0.9316 || Val Loss: 0.1641 | Val Acc: 0.9286
Epoch [04/20] Train Loss: 0.1411 | Train Acc: 0.9495 || Val Loss: 0.5466 | Val Acc: 0.8323
Epoch [05/20] Train Loss: 0.1955 | Train Acc: 0.9263 || Val Loss: 0.1374 | Val Acc: 0.9503
Epoch [06/20] Train Loss: 0.0704 | Train Acc: 0.9728 || Val Loss: 0.1551 | Val Acc: 0.9441
Epoch [07/20] Train Loss: 0.0865 | Train Acc: 0.9681 || Val Loss: 0.1934 | Val Acc: 0.9410
Epoch [08/20] Train Loss: 0.1326 | Train Acc: 0.9595 || Val Loss: 0.1589 | Val Acc: 0.9379

⛔ Early stopping triggered. Best epoch: 5


=== Evaluation Complete ===
{
  "accuracy": 0.9691358024691358,
  "macro_recall": 0.9194444444444444,
  "weighted_recall": 0.9691358024691358,
  "macro_f1": 0.9447729278812368,
  "num_classes": 3,
  "classes": [
    "Early_blight",
    "Late_blight",
    "healthy"
  ]
}

Confusion Matrix:
[[147   3   0]
 [  2 148   0]
 [  0   5  19]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant potato
=== Class-wise Recall ===
Early_blight                  : 0.9800
Late_blight                   : 0.9867
healthy                       : 0.7917
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant potato
=== Logit Margin Analysis ===
Mean margin: 7.3911
Std margin : 4.1421
Percentiles:
  10%: 2.0065
  25%: 4.5829
  50%: 7.6675
  75%: 10.2899
  90%: 12.6231


🥔 Potato — MODEL-2 Gözlemleri (Multi-class, 3 sınıf)
Potato bitkisi için MODEL-2, hastalık sınıflarını ayırt etmede yüksek doğruluk sergilemiş; ancak sağlıklı sınıf özelinde daha kırılgan bir karar sınırı ortaya koymuştur. Eğitim süreci hızlı yakınsamış, validation loss dalgalanmalarına rağmen 5. epoch’ta en iyi performans elde edilerek early stopping doğru bir noktada tetiklenmiştir. Bu durum, modelin kısa sürede doygunluğa ulaştığını göstermektedir.
Test seti üzerinde %96.9 accuracy elde edilmiş olmasına karşın, macro recall değeri 0.92 seviyesinde kalmıştır. Bu fark, sınıflar arasında performans dengesizliğine işaret etmektedir. Sınıf bazlı incelemede Early_blight (0.98) ve Late_blight (0.99) sınıflarında son derece yüksek recall değerleri elde edilirken, healthy sınıfında recall değeri 0.79 olarak ölçülmüştür. Confusion matrix incelendiğinde, sağlıklı yaprakların bir kısmının özellikle Late_blight sınıfına yanlış atandığı görülmektedir.
Bu durum, Potato özelinde hastalıklı yaprakların görsel olarak daha belirgin lezyonlar taşımasına karşın, sağlıklı yaprakların zaman zaman arka plan, ışık koşulları veya hafif doku varyasyonları nedeniyle hastalıklı sınıflara yakın bir temsil oluşturduğunu düşündürmektedir. Model, hastalık sınıflarını “pozitif” olarak yakalamakta başarılı, ancak sağlıklıyı dışlama (negative class separation) konusunda daha temkinli davranmaktadır.
Margin analizi bu yorumu desteklemektedir. Ortalama logit margin ≈ 7.39 olup, Apple ile benzer seviyededir. Ancak alt yüzde 10’luk dilimde margin değerlerinin görece düşük olması, özellikle healthy örneklerde modelin karar güveninin azaldığını göstermektedir. Sonuç olarak Potato, MODEL-2’nin hastalık sınıflarında güçlü, ancak sağlıklı sınıf ayrımında daha hassas davrandığı bir örnek olarak değerlendirilebilir.