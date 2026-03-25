

##  Beklenen Kazanımlar

- Accuracy: Büyük artış beklenmemektedir (≈ %91–92)
- **Macro Recall: artış**
- **Macro F1-score: artış**
- Validation loss: daha stabil
- Confusion matrix:
  - Early/Late blight karışıklığının azalması





sonuçlar 10 epoch için


Epoch [1/10] Train Loss: 724.0973, Train Acc: 0.3817 | Val Loss: 115.4061, Val Acc: 0.5833
Current LR: 0.000100
Epoch [2/10] Train Loss: 586.4937, Train Acc: 0.4924 | Val Loss: 107.2169, Val Acc: 0.5465
Current LR: 0.000100
Epoch [3/10] Train Loss: 528.2491, Train Acc: 0.5288 | Val Loss: 85.7516, Val Acc: 0.6697
Current LR: 0.000100
Epoch [4/10] Train Loss: 486.3362, Train Acc: 0.5597 | Val Loss: 79.4276, Val Acc: 0.7139
Current LR: 0.000100
Epoch [5/10] Train Loss: 459.6946, Train Acc: 0.5756 | Val Loss: 80.5268, Val Acc: 0.6650
Current LR: 0.000100
Epoch [6/10] Train Loss: 429.0269, Train Acc: 0.6071 | Val Loss: 73.5369, Val Acc: 0.6878
Current LR: 0.000100
Epoch [7/10] Train Loss: 414.2744, Train Acc: 0.6144 | Val Loss: 69.8103, Val Acc: 0.7164
Current LR: 0.000100
Epoch [8/10] Train Loss: 406.0378, Train Acc: 0.6157 | Val Loss: 74.7372, Val Acc: 0.6826
Current LR: 0.000100
Epoch [9/10] Train Loss: 389.2786, Train Acc: 0.6240 | Val Loss: 60.5973, Val Acc: 0.7683
Current LR: 0.000100
Epoch [10/10] Train Loss: 387.8841, Train Acc: 0.6406 | Val Loss: 58.8490, Val Acc: 0.7786
Current LR: 0.000100
Model saved to experiments/cnn/results/02_optimized/cnn_02_optimized_model.pth

===== Test Set Evaluation =====
Accuracy : 0.7663
Precision: 0.7774
Recall   : 0.7549
F1-score : 0.7501

Confusion Matrix:
[[319   0   0   0   0   0   0   1   0   0]
 [ 76  31  29   1   1   4   7   0   1   0]
 [ 30  27 225   0   3   0   0   2   0   0]
 [  3   4  15 118   3   0   1   0   0   0]
 [  5   3  24  14 203   1   5   4   6   2]
 [  8   4   7   6   5 192  26   4   0   0]
 [ 19   0   2   0   7  11 170   0   1   2]
 [213   6   4   0   7   8   0 567   0   0]
 [  0   0   1   1  11   0   0   0  44   0]
 [  1   0   2   0   0   1  10   0   0 226]]



 --------------------

 aynı modelin 20 epoch sonucu

 Epoch [1/20] Train Loss: 724.0973, Train Acc: 0.3817 | Val Loss: 115.4061, Val Acc: 0.5833
Current LR: 0.000100
Epoch [2/20] Train Loss: 586.4937, Train Acc: 0.4924 | Val Loss: 107.2169, Val Acc: 0.5465
Current LR: 0.000100
Epoch [3/20] Train Loss: 528.2491, Train Acc: 0.5288 | Val Loss: 85.7516, Val Acc: 0.6697
Current LR: 0.000100
Epoch [4/20] Train Loss: 486.3362, Train Acc: 0.5597 | Val Loss: 79.4276, Val Acc: 0.7139
Current LR: 0.000100
Epoch [5/20] Train Loss: 459.6946, Train Acc: 0.5756 | Val Loss: 80.5268, Val Acc: 0.6650
Current LR: 0.000100
Epoch [6/20] Train Loss: 429.0269, Train Acc: 0.6071 | Val Loss: 73.5369, Val Acc: 0.6878
Current LR: 0.000100
Epoch [7/20] Train Loss: 414.2744, Train Acc: 0.6144 | Val Loss: 69.8103, Val Acc: 0.7164
Current LR: 0.000100
Epoch [8/20] Train Loss: 406.0378, Train Acc: 0.6157 | Val Loss: 74.7372, Val Acc: 0.6826
Current LR: 0.000100
Epoch [9/20] Train Loss: 389.2786, Train Acc: 0.6240 | Val Loss: 60.5973, Val Acc: 0.7683
Current LR: 0.000100
Epoch [10/20] Train Loss: 387.8841, Train Acc: 0.6406 | Val Loss: 58.8490, Val Acc: 0.7786
Current LR: 0.000100
Epoch [11/20] Train Loss: 385.1905, Train Acc: 0.6348 | Val Loss: 64.5014, Val Acc: 0.7278
Current LR: 0.000100
Epoch [12/20] Train Loss: 364.8138, Train Acc: 0.6475 | Val Loss: 62.4079, Val Acc: 0.7326
Current LR: 0.000100
Epoch [13/20] Train Loss: 358.0302, Train Acc: 0.6541 | Val Loss: 51.7212, Val Acc: 0.7940
Current LR: 0.000100
Epoch [14/20] Train Loss: 348.8140, Train Acc: 0.6618 | Val Loss: 59.3364, Val Acc: 0.7624
Current LR: 0.000100
Epoch [15/20] Train Loss: 343.7090, Train Acc: 0.6577 | Val Loss: 47.0769, Val Acc: 0.8220
Current LR: 0.000100
Epoch [16/20] Train Loss: 349.4385, Train Acc: 0.6614 | Val Loss: 59.9614, Val Acc: 0.7613
Current LR: 0.000100
Epoch [17/20] Train Loss: 334.2547, Train Acc: 0.6710 | Val Loss: 54.9076, Val Acc: 0.7742
Current LR: 0.000100
Epoch [18/20] Train Loss: 329.0105, Train Acc: 0.6758 | Val Loss: 53.8313, Val Acc: 0.7830
Current LR: 0.000030
Epoch [19/20] Train Loss: 318.4376, Train Acc: 0.6851 | Val Loss: 44.8701, Val Acc: 0.8242
Current LR: 0.000030
Epoch [20/20] Train Loss: 317.7462, Train Acc: 0.6858 | Val Loss: 48.1700, Val Acc: 0.8139
Current LR: 0.000030

===== Test Set Evaluation =====
Accuracy : 0.7944
Precision: 0.8036
Recall   : 0.7990
F1-score : 0.7886

Confusion Matrix:
[[320   0   0   0   0   0   0   0   0   0]
 [ 54  53  25   1   1   0  14   1   1   0]
 [ 20  34 226   3   3   0   1   0   0   0]
 [  1   1  10 124   6   0   2   0   0   0]
 [  3   6  13   5 216   1  14   0   7   2]
 [  2   4   4   3   3 199  33   4   0   0]
 [ 11   0   1   0   2   6 190   0   0   2]
 [199  18   2   5   4   3   0 574   0   0]
 [  0   0   1   1   5   0   1   0  49   0]
 [  1   0   1   0   0   0  17   0   0 221]]


## 02_Optimized CNN (20 Epoch) – Ek Çalıştırma Sonrası Çıkarımlar

02_optimized CNN modeli, ilk etapta 10 epoch ile eğitilmişti. Ancak eğitim ve validation eğrilerinde hâlâ yukarı yönlü bir trend gözlemlenmesi üzerine, modelin gerçekten doyuma ulaşıp ulaşmadığını anlamak amacıyla **aynı mimari ve aynı ayarlar korunarak** epoch sayısı **20’ye çıkarılmıştır**. Bu çalışma bir optimizasyon değil, **tanısal (diagnostic) bir test** olarak ele alınmıştır.

---

### Eğitim Davranışı (20 Epoch Gözlemi)

- Train accuracy: **%38 → %69**
- Validation accuracy: **%58 → %82 (epoch 15 civarı)**
- Overfitting belirtisi gözlemlenmemiştir.
- Train–validation farkı kontrollü ve stabildir.
- ReduceLROnPlateau scheduler **epoch 18’de** tetiklenmiş ve learning rate düşürülmüştür.

Bu davranış, modelin ilk 10 epoch’ta underfit olduğu yönündeki şüpheleri **çürütmüştür**. Modelin aslında öğrenme kapasitesine sahip olduğu, ancak **erken kesildiği** net biçimde görülmüştür.

---

### Test Set Sonuçları (20 Epoch)

- Accuracy: **0.7944**
- Macro Precision: **0.8036**
- Macro Recall: **0.7990**
- Macro F1-score: **0.7886**

Bu sonuçlar, 10 epoch’luk 02 çalışmasına kıyasla **anlamlı bir iyileşme** göstermektedir. Özellikle macro recall ve F1-score değerlerinin yükselmesi, modelin sınıflar arası daha dengeli davrandığını doğrulamaktadır.

---

### Confusion Matrix Üzerinden Gözlemler

- Küçük sınıflar tamamen ezilmemiştir; model her sınıfa tahmin üretmektedir.
- Büyük sınıflar (özellikle Tomato_Yellow_Leaf_Curl_Virus ve Bacterial Spot) hâlâ belirgin şekilde cezalandırılmaktadır.
- Bazı sınıflar arası karışmalar (ör. TYLCV ↔ Bacterial Spot) devam etmektedir.

Bu durum, epoch sayısının artırılmasının **öğrenmeyi geliştirdiğini**, ancak **class weight agresifliğinin** hâlâ temel problem olarak kaldığını göstermektedir.

---

### Temel Çıkarımlar

Bu ek çalıştırma sonucunda aşağıdaki net sonuçlara ulaşılmıştır:

1. 02_optimized CNN modeli underfit değildir; yalnızca erken kesilmiştir.
2. Epoch sayısının artırılması performansı anlamlı biçimde iyileştirmiştir.
3. Learning rate ve scheduler mekanizması doğru çalışmaktadır.
4. Buna rağmen:
   - Class weight yaklaşımı hâlâ fazla agresiftir.
   - Convolutional dropout, feature learning’i gereğinden fazla sınırlamaktadır.
5. Yani problem **zaman değil**, **ceza ve regularization dengesidir**.

---

### Sonuç

02_optimized CNN’in 20 epoch’luk eğitimi, modelin kapasite sınırlarını ve darboğazlarını net biçimde ortaya koymuştur. Bu çalışma, 03_experimental aşamasına geçmeden önce alınmış **bilinçli bir ara karar** niteliğindedir. Elde edilen bulgular, bir sonraki deneyde yalnızca gerekli kısıtların gevşetilmesini sağlayarak daha hedefli bir optimizasyon yapılmasına olanak tanımıştır.
