# KNN Model Analizi  
## Tomato Leaf Disease Classification

Bu doküman, Tomato Leaf Disease veri seti üzerinde eğitilen **Basic KNN (01)** ve **Optimized KNN (02)** modellerinin performans karşılaştırmasını, KNN algoritmasının bu problemde neden sınırlı kaldığını ve **Optimized KNN’de train accuracy’nin neden %100 çıktığının teknik açıklamasını** içermektedir.

---

## 1. Problem Tanımı

- Görev: Çok sınıflı bitki hastalığı sınıflandırması (10 sınıf)
- Girdi: 64×64 boyutlu RGB görüntüler  
- Özellik uzayı: 12 288 boyutlu düzleştirilmiş piksel vektörü
- Veri bölünmesi: Train / Validation / Test (stratified)

Amaç:
- KNN algoritmasının görüntü verisi üzerindeki davranışını gözlemlemek
- Baseline → Optimized evrimini göstermek
- CNN’e geçiş için deneysel gerekçe oluşturmak

---

## 2. KNN 01 – Basic (Baseline)

### Model Yapılandırması
- `n_neighbors = 5`
- `weights = uniform`
- Mesafe metriği: Euclidean
- **Feature scaling uygulanmadı** (bilinçli tercih)

### Genel Performans

| Veri Seti | Accuracy |
|----------|----------|
| Train    | 0.6998   |
| Validation | 0.6249 |
| Test     | 0.6222   |

### Temel Gözlemler

- KNN tamamen mesafe tabanlı bir algoritmadır.
- Ölçeklendirme olmadığı için piksel uzayındaki mesafeler tutarsızdır.
- Training set üzerinde bile bazı örnekler:
  - Kendilerine en yakın komşu olarak **yanlış sınıftaki örnekleri** seçebilmektedir.
- Özellikle **Early Blight** ve **Tomato Mosaic Virus** sınıflarında recall çok düşüktür.
- **Tomato Yellow Leaf Curl Virus**, görsel olarak ayırt edici olduğu için yüksek performans göstermiştir.

---

## 3. KNN 02 – Optimized

### Yapılan İyileştirmeler
- **StandardScaler** eklendi
- `weights = distance` kullanıldı
- `k` için hiperparametre araması yapıldı (3–11)

### Hyperparameter Arama Özeti

En iyi sonuç (validation bazlı):
- **k = 8**
- Validation Accuracy: **0.6506**
- Test Accuracy: **0.6448**

### Seçilen Modelin Performansı

| Veri Seti | Accuracy |
|----------|----------|
| Train    | **1.0000** |
| Validation | 0.6506 |
| Test     | 0.6448 |

---

## 4. Neden Optimized KNN’de Train Accuracy %100?

Optimized KNN’de training accuracy’nin %100 çıkması **normal ve beklenen** bir durumdur.

### 4.1 KNN’in Instance-Based Yapısı

- KNN parametre öğrenmez.
- Training örneklerini doğrudan bellekte tutar.
- Tahmin sırasında:
  > “Bu örneğe en yakın `k` örnek hangileri?”  
  sorusunu sorar.

Bu nedenle KNN, **ezberci (lazy learner)** bir algoritmadır.

---

### 4.2 StandardScaler Etkisi

Optimized modelde:
- Tüm feature’lar aynı ölçeğe getirilmiştir.
- Piksel uzayındaki mesafeler anlamlı hale gelmiştir.

Sonuç:
- Bir training örneği, kendisine **en yakın nokta olarak çoğu zaman kendisini** seçer.
- Mesafe tabanlı algoritmalar için bu durum ezberi kolaylaştırır.

---

### 4.3 Distance-Weighted Voting Etkisi

`weights = distance` kullanıldığında:
- Yakın komşular yüksek ağırlık alır
- Uzak komşuların etkisi azalır

Training set için:
- En yakın komşu genellikle **aynı örnek** veya **çok benzeri** olduğu için
- Doğru sınıfın oyu baskın gelir

➡️ Training accuracy %100 olur.

---

### 4.4 Yüksek Boyutun Rolü

- 12 288 boyutlu uzayda her training örneği çok spesifik bir noktadadır.
- KNN bu uzayda genelleme yapmak zorunda değildir.
- Ezber davranışı daha da güçlenir.

---

## 5. Peki Neden KNN 01’de Train %100 Değildi?

Bu da tamamen mantıklıdır.

### 5.1 Ölçeklendirme Yoktu
- Ham piksel değerlerinde bazı feature’lar mesafeyi domine eder.
- Bir training örneği:
  - Kendisine en yakın örnek olarak **başka sınıftan** bir görüntüyü seçebilir.

### 5.2 Uniform Voting Kullanıldı
- Yakın ve uzak komşular eşit oy alır.
- Uzak ama yanlış sınıftaki komşular:
  - Sonucu bozabilir.

### 5.3 Sonuç
- KNN 01:
  - Ne tam ezberleyebilmiş
  - Ne de iyi genelleştirebilmiştir
- Bu yüzden training accuracy %70 civarında kalmıştır.

---

## 6. KNN 01 vs KNN 02 – Karşılaştırma

| Model | Validation Acc | Test Acc |
|------|----------------|----------|
| KNN 01 (Basic) | 0.6249 | 0.6222 |
| KNN 02 (Optimized) | **0.6506** | **0.6448** |

### Yorum
- Ölçeklendirme ve distance-weighted voting ile **~%2–3 mutlak iyileşme** sağlanmıştır.
- Ancak iyileşme sınırlıdır ve KNN’in teorik sınırlarına ulaşılmıştır.

---

## 7. KNN Bu Problemde Neden Başarısız Oldu?

### Teknik Nedenler

1. **Curse of Dimensionality**
   - Yüksek boyutlu uzayda mesafeler anlamsızlaşır.

2. **Feature Learning Yok**
   - KNN ham piksellerle çalışır.
   - Kenar, doku, şekil gibi semantik bilgileri öğrenemez.

3. **Görsel Olarak Benzer Sınıflar**
   - Early Blight ve Mosaic Virus gibi hastalıklar birbirine çok benzer.

4. **Instance-Based Yapı**
   - Genelleme yerine ezber ön plandadır.



## 9. Sonuç

KNN deneyleri:
- Bilinçli olarak baseline ve optimized şeklinde kurgulanmıştır
- Algoritmanın güçlü ve zayıf yönleri net biçimde ortaya konmuştur
- CNN ve Transfer Learning’e geçiş için sağlam bir deneysel zemin oluşturmuştur


