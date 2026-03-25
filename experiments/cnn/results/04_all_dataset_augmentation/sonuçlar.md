# 04 – Tüm Dataset Üzerinde Augmentation Deneyi

## 📌 Deney Özeti

**Proje:** Plant Disease Classification  
**Dataset:** PlantVillage – Full Dataset (38 sınıf)  
**Model:** CNN_04_All_Dataset_Augmentation  
**Eğitim Türü:** From-scratch CNN  
**Epoch:** 30  
**Optimizer:** Adam (lr = 1e-4)  
**Scheduler:** ReduceLROnPlateau  
**Loss:** Softened class weight’li CrossEntropyLoss  
**Cihaz:** Apple Silicon (MPS)

Bu deneyde amaç, **online data augmentation** kullanımının,  
PlantVillage full dataset üzerinde from-scratch eğitilen bir CNN’in  
genelleme performansını ve özellikle küçük sınıflardaki başarımını artırıp artırmadığını incelemektir.

---

## 🔧 Yapılan Çalışma

### 1️⃣ Motivasyon
Önceki deneyde (03_all_dataset – augmentation yok):
- ~0.92 test accuracy
- ~0.89 macro F1-score

elde edilmiştir.  
Ancak küçük sınıflardaki **recall** değerlerini daha da iyileştirme ihtimali nedeniyle
augmentation denenmiştir.

Hipotez:
- Eğitim verisine çeşitlilik kazandırmak
- CNN embedding’lerini daha dayanıklı hale getirmek
- İleride CNN → SVM pipeline’ı için daha iyi temsil öğrenmek

---

### 2️⃣ Augmentation Stratejisi (Sadece Train Set)

Augmentation **yalnızca train set** üzerinde uygulanmıştır.  
Validation ve test setleri **değiştirilmemiştir**.

**Kullanılan dönüşümler:**
- RandomHorizontalFlip (p = 0.5)
- RandomRotation (±15 derece)
- RandomResizedCrop (scale = 0.9 – 1.0)

**Bilinçli olarak kullanılmayanlar:**
- Shear
- Vertical flip
- Büyük açılı rotasyonlar
- Agresif crop
- Color jitter

Amaç, **biyolojik olarak mantıklı ve sınırlı** bir augmentation uygulamaktır.

---

### 3️⃣ Regularization Yapısı

Bu deneyde aynı anda şu regularization yöntemleri kullanılmıştır:
- Data augmentation
- Classifier dropout
- Softened class weights (sqrt(1 / count))

Bu durum model üzerinde **yüksek seviyede regularization** etkisi oluşturmuştur.

---

## 📊 Sonuçlar

### 🔹 Eğitim Süreci Gözlemleri
- Train accuracy yaklaşık **%54** civarında plato yaptı
- Validation accuracy düzenli olarak yükseldi (~%87)
- Learning rate anlamlı biçimde düşmedi
- Bu tablo **overfitting değil, underfitting** işaretidir

---

### 🔹 Test Set Sonuçları

| Metrik | Değer |
|------|------|
| Accuracy | **0.8763** |
| Macro Precision | **0.8456** |
| Macro Recall | **0.8228** |
| Macro F1-score | **0.8236** |

---

### 🔹 Augmentation Olmayan Model ile Karşılaştırma

| Model | Accuracy | Macro F1 |
|------|----------|----------|
| 03 – Augmentation Yok (30 ep) | **0.9230** | **0.8940** |
| 04 – Augmentation Var (30 ep) | **0.8763** | **0.8236** |

Augmentation, tüm metriklerde **belirgin performans düşüşüne** yol açmıştır.

---

## 🧠 Çıkarımlar

1. **Augmentation bu dataset için fayda sağlamadı**
   - Accuracy ve macro F1 anlamlı şekilde düştü
   - Küçük sınıflarda beklenen recall artışı gözlenmedi

2. **Dataset yapısı belirleyici oldu**
   - PlantVillage görüntüleri:
     - Merkezlenmiş
     - Temiz
     - Arka planı kontrol altında
   - Augmentation, hastalık bölgelerini kısmen bozarak
     ayırt edici lokal desenlerin öğrenilmesini zorlaştırdı

3. **Aşırı regularization etkisi**
   - From-scratch CNN
   - Class weight
   - Dropout
   - Augmentation

   Bu kombinasyon, modelin yeterince öğrenmesini engelleyerek
   **underfitting** oluşturdu.

4. **Train accuracy tek başına güvenilir değil**
   - Augmentation altında düşük train accuracy normaldir
   - Ancak test sonuçları, temsil öğreniminin zayıfladığını doğruladı

---

## 🎯 Sonuç

> PlantVillage full dataset üzerinde, from-scratch CNN eğitimi için
> data augmentation performansı artırmamış, aksine düşürmüştür.
> Datasetin zaten temiz ve normalize yapısı nedeniyle augmentation,
> ayırt edici hastalık desenlerini zayıflatmış ve underfitting’e yol açmıştır.

Bu deney göstermiştir ki:
- Augmentation her zaman faydalı değildir
- Dataset’e özgü yapı mutlaka dikkate alınmalıdır
- Standart teknikler ampirik olarak test edilmeden kabul edilmemelidir

---

## 🚀 Sonraki Adımlar

- Augmentation yaklaşımı bu proje için sonlandırılmıştır
- **03_all_dataset (augmentation yok)** modeli referans alınacaktır
- Bu modelden elde edilen embedding’ler ile **CNN → SVM** denemesi yapılacaktır
- Gerekirse bir sonraki aşamada **transfer learning** değerlendirilecektir

---

## 💬 Mülakat İçin Özet Cümle

> “PlantVillage full dataset üzerinde augmentation denedim ancak datasetin zaten temiz
> ve merkezli yapısı nedeniyle performansı düşürdüğünü gözlemledim.
> Bu yüzden augmentation’dan bilinçli olarak vazgeçip,
> daha güçlü olan non-augmented CNN modelini feature extractor olarak kullanmaya karar verdim.”

Using device: mps
Epoch [1/30] Train Loss: 3358.1614, Train Acc: 0.2979 | Val Loss: 469.7861, Val Acc: 0.6071
Current LR: 0.000100
Epoch [2/30] Train Loss: 2776.5559, Train Acc: 0.3969 | Val Loss: 339.5672, Val Acc: 0.6484
Current LR: 0.000100
Epoch [3/30] Train Loss: 2563.5998, Train Acc: 0.4248 | Val Loss: 302.9461, Val Acc: 0.7082
Current LR: 0.000100
Epoch [4/30] Train Loss: 2421.8406, Train Acc: 0.4413 | Val Loss: 258.9240, Val Acc: 0.7292
Current LR: 0.000100
Epoch [5/30] Train Loss: 2323.6251, Train Acc: 0.4547 | Val Loss: 238.8764, Val Acc: 0.7633
Current LR: 0.000100
Epoch [6/30] Train Loss: 2274.8609, Train Acc: 0.4604 | Val Loss: 235.9796, Val Acc: 0.7637
Current LR: 0.000100
Epoch [7/30] Train Loss: 2208.4458, Train Acc: 0.4713 | Val Loss: 216.8903, Val Acc: 0.7746
Current LR: 0.000100
Epoch [8/30] Train Loss: 2173.7902, Train Acc: 0.4785 | Val Loss: 223.2890, Val Acc: 0.7810
Current LR: 0.000100
Epoch [9/30] Train Loss: 2135.7753, Train Acc: 0.4842 | Val Loss: 199.9959, Val Acc: 0.7944
Current LR: 0.000100
Epoch [10/30] Train Loss: 2087.3311, Train Acc: 0.4924 | Val Loss: 197.9514, Val Acc: 0.8026
Current LR: 0.000100
Epoch [11/30] Train Loss: 2056.9348, Train Acc: 0.4962 | Val Loss: 190.3527, Val Acc: 0.7858
Current LR: 0.000100
Epoch [12/30] Train Loss: 2031.6960, Train Acc: 0.4969 | Val Loss: 188.2042, Val Acc: 0.8086
Current LR: 0.000100
Epoch [13/30] Train Loss: 2014.3890, Train Acc: 0.4977 | Val Loss: 167.8618, Val Acc: 0.8288
Current LR: 0.000100
Epoch [14/30] Train Loss: 1983.8099, Train Acc: 0.5051 | Val Loss: 168.6243, Val Acc: 0.8199
Current LR: 0.000100
Epoch [15/30] Train Loss: 1965.8079, Train Acc: 0.5095 | Val Loss: 165.6063, Val Acc: 0.8217
Current LR: 0.000100
Epoch [16/30] Train Loss: 1949.4109, Train Acc: 0.5152 | Val Loss: 161.3746, Val Acc: 0.8328
Current LR: 0.000100
Epoch [17/30] Train Loss: 1933.3105, Train Acc: 0.5185 | Val Loss: 163.7467, Val Acc: 0.8417
Current LR: 0.000100
Epoch [18/30] Train Loss: 1913.7242, Train Acc: 0.5160 | Val Loss: 155.5799, Val Acc: 0.8360
Current LR: 0.000100
Epoch [19/30] Train Loss: 1908.0314, Train Acc: 0.5201 | Val Loss: 156.9144, Val Acc: 0.8370
Current LR: 0.000100
Epoch [20/30] Train Loss: 1880.2314, Train Acc: 0.5280 | Val Loss: 149.8358, Val Acc: 0.8551
Current LR: 0.000100
Epoch [21/30] Train Loss: 1871.2081, Train Acc: 0.5246 | Val Loss: 147.3188, Val Acc: 0.8488
Current LR: 0.000100
Epoch [22/30] Train Loss: 1866.6824, Train Acc: 0.5278 | Val Loss: 137.8861, Val Acc: 0.8556
Current LR: 0.000100
Epoch [23/30] Train Loss: 1853.9160, Train Acc: 0.5310 | Val Loss: 148.9180, Val Acc: 0.8341
Current LR: 0.000100
Epoch [24/30] Train Loss: 1838.3124, Train Acc: 0.5303 | Val Loss: 136.1628, Val Acc: 0.8631
Current LR: 0.000100
Epoch [25/30] Train Loss: 1827.7455, Train Acc: 0.5366 | Val Loss: 141.4611, Val Acc: 0.8524
Current LR: 0.000100
Epoch [26/30] Train Loss: 1823.4609, Train Acc: 0.5355 | Val Loss: 128.8453, Val Acc: 0.8694
Current LR: 0.000100
Epoch [27/30] Train Loss: 1800.9222, Train Acc: 0.5374 | Val Loss: 122.4307, Val Acc: 0.8648
Current LR: 0.000100
Epoch [28/30] Train Loss: 1797.4615, Train Acc: 0.5393 | Val Loss: 132.5154, Val Acc: 0.8632
Current LR: 0.000100
Epoch [29/30] Train Loss: 1794.9643, Train Acc: 0.5422 | Val Loss: 133.7570, Val Acc: 0.8508
Current LR: 0.000100
Epoch [30/30] Train Loss: 1783.2452, Train Acc: 0.5434 | Val Loss: 119.5013, Val Acc: 0.8748
Current LR: 0.000100



===== Test Set Evaluation (04 All Dataset Augmentation) =====
Accuracy : 0.8763
Precision: 0.8456
Recall   : 0.8228
F1-score : 0.8236

Confusion Matrix:
[[ 56   2   0 ...   0   0   0]
 [  0  86   0 ...   0   0   0]
 [  0   0  21 ...   0   0   0]
 ...
 [  0   0   0 ... 782   0   0]
 [  0   0   0 ...   0  54   0]