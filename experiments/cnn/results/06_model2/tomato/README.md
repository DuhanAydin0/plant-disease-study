=== Evaluation Complete ===
{
  "accuracy": 0.8811265544989028,
  "macro_recall": 0.8357109910231248,
  "weighted_recall": 0.8811265544989028,
  "macro_f1": 0.8430442427519994,
  "num_classes": 10,
  "classes": [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites_Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy"
  ]
}

Confusion Matrix:
[[297   1   6   0   0   0   1  14   0   1]
 [ 24  74  24   1   5   1  12   7   0   2]
 [  5  15 239   5   7   3   0  10   0   3]
 [  0   5   3 112  16   5   1   2   0   0]
 [  0   2  16   3 223   1   9   5   7   1]
 [  0   1   2   0   8 209  17  11   3   1]
 [  2   3   1   0  11   8 184   0   0   3]
 [  9   2   1   0   1   5   0 786   0   1]
 [  0   0   0   1   5   4   0   0  47   0]
 [  1   0   1   0   0   0   0   0   0 238]]
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/recall_analysis.py --plant tomato
=== Class-wise Recall ===
Bacterial_spot                : 0.9281
Early_blight                  : 0.4933
Late_blight                   : 0.8328
Leaf_Mold                     : 0.7778
Septoria_leaf_spot            : 0.8352
Spider_mites_Two-spotted_spider_mite: 0.8294
Target_Spot                   : 0.8679
Tomato_Yellow_Leaf_Curl_Virus : 0.9764
Tomato_mosaic_virus           : 0.8246
healthy                       : 0.9917
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % python experiments/cnn/runs/06_model2/margin_analysis.py --plant tomato
=== Logit Margin Analysis ===
Mean margin: 3.5098
Std margin : 3.3335
Percentiles:
  10%: -0.3586
  25%: 1.2842
  50%: 3.2479
  75%: 5.3675
  90%: 7.8226
(.venv) duhanaydin@Duhan-MacBook-Air plant disease study % 





==============================
MODEL-2 TRAINING STARTED
Plant: tomato
Classes (10): ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites_Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy']
Device: mps
==============================

Epoch [01/30] Train Loss: 2.0206 | Train Acc: 0.3037 || Val Loss: 1.7392 | Val Acc: 0.3523 | LR: 1.00e-05
Epoch [02/30] Train Loss: 1.5679 | Train Acc: 0.4739 || Val Loss: 1.3312 | Val Acc: 0.6160 | LR: 1.00e-05
Epoch [03/30] Train Loss: 1.2802 | Train Acc: 0.5953 || Val Loss: 1.0938 | Val Acc: 0.6969 | LR: 1.00e-05
Epoch [04/30] Train Loss: 1.1172 | Train Acc: 0.6467 || Val Loss: 0.9673 | Val Acc: 0.7212 | LR: 1.00e-05
Epoch [05/30] Train Loss: 1.0067 | Train Acc: 0.6845 || Val Loss: 0.8556 | Val Acc: 0.7451 | LR: 1.00e-05
Epoch [06/30] Train Loss: 0.9330 | Train Acc: 0.7057 || Val Loss: 0.7861 | Val Acc: 0.7529 | LR: 1.00e-05
Epoch [07/30] Train Loss: 0.8757 | Train Acc: 0.7198 || Val Loss: 0.7322 | Val Acc: 0.7779 | LR: 1.00e-05
Epoch [08/30] Train Loss: 0.8238 | Train Acc: 0.7326 || Val Loss: 0.7011 | Val Acc: 0.7760 | LR: 1.00e-05
Epoch [09/30] Train Loss: 0.7901 | Train Acc: 0.7468 || Val Loss: 0.6577 | Val Acc: 0.7974 | LR: 1.00e-05
Epoch [10/30] Train Loss: 0.7532 | Train Acc: 0.7555 || Val Loss: 0.6446 | Val Acc: 0.7996 | LR: 1.00e-05
Epoch [11/30] Train Loss: 0.7227 | Train Acc: 0.7638 || Val Loss: 0.6045 | Val Acc: 0.8084 | LR: 1.00e-05
Epoch [12/30] Train Loss: 0.7009 | Train Acc: 0.7729 || Val Loss: 0.5853 | Val Acc: 0.8187 | LR: 1.00e-05
Epoch [13/30] Train Loss: 0.6649 | Train Acc: 0.7795 || Val Loss: 0.5626 | Val Acc: 0.8213 | LR: 1.00e-05
Epoch [14/30] Train Loss: 0.6468 | Train Acc: 0.7852 || Val Loss: 0.5380 | Val Acc: 0.8290 | LR: 1.00e-05
Epoch [15/30] Train Loss: 0.6234 | Train Acc: 0.7953 || Val Loss: 0.5250 | Val Acc: 0.8231 | LR: 1.00e-05
Epoch [16/30] Train Loss: 0.6115 | Train Acc: 0.7946 || Val Loss: 0.5112 | Val Acc: 0.8352 | LR: 1.00e-05
Epoch [17/30] Train Loss: 0.5912 | Train Acc: 0.8070 || Val Loss: 0.4876 | Val Acc: 0.8419 | LR: 1.00e-05
Epoch [18/30] Train Loss: 0.5776 | Train Acc: 0.8144 || Val Loss: 0.4937 | Val Acc: 0.8389 | LR: 1.00e-05
Epoch [19/30] Train Loss: 0.5521 | Train Acc: 0.8156 || Val Loss: 0.4714 | Val Acc: 0.8441 | LR: 1.00e-05
Epoch [20/30] Train Loss: 0.5452 | Train Acc: 0.8216 || Val Loss: 0.4552 | Val Acc: 0.8455 | LR: 1.00e-05
Epoch [21/30] Train Loss: 0.5383 | Train Acc: 0.8214 || Val Loss: 0.4551 | Val Acc: 0.8499 | LR: 1.00e-05
Epoch [22/30] Train Loss: 0.5162 | Train Acc: 0.8284 || Val Loss: 0.4359 | Val Acc: 0.8522 | LR: 1.00e-05
Epoch [23/30] Train Loss: 0.5087 | Train Acc: 0.8288 || Val Loss: 0.4370 | Val Acc: 0.8536 | LR: 1.00e-05
Epoch [24/30] Train Loss: 0.4944 | Train Acc: 0.8362 || Val Loss: 0.4212 | Val Acc: 0.8621 | LR: 1.00e-05
Epoch [25/30] Train Loss: 0.4854 | Train Acc: 0.8395 || Val Loss: 0.4131 | Val Acc: 0.8647 | LR: 1.00e-05
Epoch [26/30] Train Loss: 0.4754 | Train Acc: 0.8406 || Val Loss: 0.4135 | Val Acc: 0.8617 | LR: 1.00e-05
Epoch [27/30] Train Loss: 0.4654 | Train Acc: 0.8417 || Val Loss: 0.4017 | Val Acc: 0.8639 | LR: 1.00e-05
Epoch [28/30] Train Loss: 0.4540 | Train Acc: 0.8464 || Val Loss: 0.4008 | Val Acc: 0.8647 | LR: 1.00e-05
Epoch [29/30] Train Loss: 0.4408 | Train Acc: 0.8524 || Val Loss: 0.3896 | Val Acc: 0.8658 | LR: 1.00e-05
Epoch [30/30] Train Loss: 0.4407 | Train Acc: 0.8532 || Val Loss: 0.3819 | Val Acc: 0.8716 | LR: 1.00e-05




🍅 Tomato — MODEL-2 Gözlemleri (Multi-class, 10 sınıf)
Tomato bitkisi, MODEL-2 kapsamında en zor ve en karmaşık çok sınıflı problem olarak öne çıkmıştır. On farklı hastalık sınıfı ve yüksek sınıf içi benzerlik nedeniyle model, diğer bitkilere kıyasla daha düşük ancak anlamlı bir performans sergilemiştir. Test setinde %88.1 accuracy ve 0.84 macro recall elde edilmiştir. Accuracy ile macro recall arasındaki fark, modelin bazı sınıflarda güçlü performans gösterirken bazı sınıflarda belirgin zorlanmalar yaşadığını ortaya koymaktadır.
Sınıf bazlı recall analizi, bu dengesizliğin açık kaynaklarını göstermektedir. Tomato_Yellow_Leaf_Curl_Virus (0.98) ve healthy (0.99) sınıflarında son derece yüksek recall elde edilirken, Early_blight sınıfında recall değeri 0.49 seviyesinde kalmıştır. Confusion matrix incelendiğinde, Early_blight örneklerinin ağırlıklı olarak Late_blight, Septoria_leaf_spot ve Target_Spot sınıflarıyla karıştığı görülmektedir. Bu durum, Tomato dataset’inde bazı fungal hastalıkların leke şekli, renk ve yayılım paternleri açısından ciddi görsel örtüşme taşıdığını göstermektedir.
Genel hata yapısı incelendiğinde modelin hatalarının dağınık fakat belirli hastalık kümeleri etrafında yoğunlaştığı görülmektedir. Bacterial_spot, Late_blight, Septoria_leaf_spot ve Target_Spot sınıfları arasında çift yönlü karışmalar mevcuttur. Buna karşın viral hastalıklar (özellikle Yellow Leaf Curl Virus) diğer sınıflardan net biçimde ayrılmıştır. Bu durum, MODEL-2’nin bazı hastalık tiplerinde (viral vs. fungal) ayrımı başarıyla yapabildiğini, ancak aynı tip içindeki alt hastalıklar arasında karar sınırının zayıfladığını göstermektedir.
Margin analizi Tomato için MODEL-2’nin en zayıf karar güvenine sahip olduğu durumu ortaya koymaktadır. Ortalama logit margin ≈ 3.51 ile diğer bitkilere kıyasla belirgin şekilde düşüktür. Alt %10’luk dilimde margin değerlerinin negatif olması, modelin bu örneklerde yanlış sınıfa daha yüksek güvenle karar verdiğini göstermektedir. Bu sonuç, confusion matrix’te gözlenen çoklu sınıf karışmalarını doğrudan doğrulamaktadır.
Sonuç olarak Tomato, MODEL-2’nin problem ayrıştırmasına rağmen yüksek sınıf sayısı ve yüksek görsel benzerlik durumlarında hâlâ sınırlara sahip olduğunu göstermektedir. Bu bitki, MODEL-2 analizinde “en zor senaryo” olarak kritik bir karşı örnek sunmakta ve tek CNN yaklaşımının neden karar sınırı problemleri yaşadığını güçlü biçimde ortaya koymaktadır. 