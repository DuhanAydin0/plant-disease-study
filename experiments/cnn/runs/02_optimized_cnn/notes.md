# 02 Optimized CNN

## Purpose

This experiment implements an optimized CNN architecture with improvements over the baseline, such as better regularization, advanced optimization techniques, and architectural enhancements to improve performance on tomato leaf disease classification.





# 02_Optimized_CNN – Tomato Leaf Disease Classification

## Amaç
Bu aşamanın amacı, `01_baseline_cnn` modelinde gözlemlenen **overfitting**, **class imbalance etkisi** ve **kararsız validation loss** problemlerini,
**mimariyi büyütmeden** yalnızca **eğitim disiplini ve düzenlileştirme (regularization)** yöntemleri ile iyileştirmektir.

Bu çalışma:
- Daha derin veya karmaşık bir CNN kurmayı değil
- Aynı mimariyi **daha kontrollü ve genellenebilir** hale getirmeyi hedefler.

---

## 1. Baseline Model Özeti (01_baseline_cnn)

### Mimari
- 3 adet Convolutional blok  
  `Conv2d → ReLU → MaxPool`
- Fully Connected classifier
- Input resolution: **224 × 224**
- Optimizer: Adam (LR = 1e-3)
- Loss: CrossEntropyLoss
- Regularization: **YOK**

### Baseline Sonuçları (Test Set)
- Accuracy: **0.9115**
- Macro Precision: **0.8966**
- Macro Recall: **0.8831**
- Macro F1-score: **0.8778**

### Gözlemler
- Train accuracy hızlı şekilde %99 seviyesine ulaşmıştır.
- Validation loss epoch ilerledikçe **dalgalı ve yükselme eğilimindedir**.
- Bu durum, klasik bir **overfitting** göstergesidir.
- Confusion matrix incelendiğinde:
  - Büyük sınıfların (ör. Tomato_Yellow_Leaf_Curl_Virus) baskın olduğu
  - Veri sayısı az ve görsel olarak benzer sınıfların (Early/Late blight) karıştığı görülmüştür.

---

## 2. Tespit Edilen Temel Problemler

### 2.1 Regularization Eksikliği
Baseline CNN:
- Batch Normalization içermemektedir
- Dropout içermemektedir

Bu durum:
- Modelin çok hızlı ezberlemesine
- Validation performansının kararsızlaşmasına yol açmaktadır.

---

### 2.2 Learning Rate’in Agresif Olması
- Adam optimizer ile **1e-3** learning rate kullanılmıştır.
- 224×224 giriş boyutu ve sınırlı veri miktarı için bu değer yüksektir.
- Sonuç: Validation loss dalgalanması.

---

### 2.3 Class Imbalance Etkisi
- Sınıflar arasında belirgin örnek sayısı farkı vardır.
- CrossEntropyLoss varsayılan haliyle tüm sınıfları eşit ağırlıkta ele alır.
- Bu nedenle küçük sınıflar eğitim sırasında baskılanmaktadır.
- Macro recall ve F1-score’un accuracy’ye göre daha düşük kalmasının ana sebeplerinden biridir.

---

### 2.4 Eğitim Sürecinin Kör İlerlemesi
- Learning rate scheduler yoktur.
- Early stopping yoktur.
- Model, ne zaman “yeterince öğrendiğini” bilmemektedir.

---

## 3. Optimized CNN Yaklaşımı (02_optimized_cnn)

Bu aşamada **mimari büyütülmeyecektir**.

### Bilinçli olarak yapılmayanlar
- Daha derin CNN
- Kernel boyutu artırımı
- Transfer learning
- Oversampling

Amaç:  
**Aynı CNN’i daha disiplinli eğitmek**

---

## 4. Yapılacak Optimizasyonlar

### 4.1 Batch Normalization
Her convolutional blok şu şekilde güncellenecektir:



Conv2d → BatchNorm → ReLU → MaxPool

**Beklenen etki:**
- Daha stabil gradient akışı
- Daha düzgün feature dağılımı
- Daha kararlı validation loss

---

### 4.2 Dropout
- Convolutional bloklar sonrası: `Dropout(p=0.25)`
- Fully connected katman sonrası: `Dropout(p=0.5)`

**Beklenen etki:**
- Ezberlemenin kırılması
- Train–validation farkının azalması

---

### 4.3 Learning Rate Düşürülmesi
- Adam learning rate:
1e-3 → 1e-4





**Beklenen etki:**
- Daha kontrollü öğrenme
- Validation loss dalgalanmasının azalması

---

### 4.4 Learning Rate Scheduler
- `ReduceLROnPlateau` kullanılacaktır.
- Validation loss izlenecek,
- İyileşme durduğunda learning rate otomatik düşürülecektir.

---

### 4.5 Class Weight Kullanımı
- Loss fonksiyonuna **class_weight** eklenecektir.
- Amaç:
  - Küçük sınıfların gradient katkısını artırmak
  - Macro recall ve F1-score’u iyileştirmek
- Aşırı agresif ağırlıklandırmadan kaçınılacaktır.

---

## 5. Beklenen Kazanımlar

- Accuracy: Büyük artış beklenmemektedir (≈ %91–92)
- **Macro Recall: artış**
- **Macro F1-score: artış**
- Validation loss: daha stabil
- Confusion matrix:
  - Early/Late blight karışıklığının azalması

Bu iyileştirmeler, modeli:
- CV için daha güçlü
- Mülakatlarda daha savunulabilir
- Gerçek dünyaya daha uygun hale getirecektir.

---

## 6. Sonraki Adım
Bu dokümanın ardından:
- `02_optimized_cnn` kodu adım adım yazılacak
- Her değişikliğin baseline’a göre etkisi gözlemlenecektir



