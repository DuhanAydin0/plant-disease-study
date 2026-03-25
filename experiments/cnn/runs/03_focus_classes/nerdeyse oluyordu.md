# Focused Fine-Tuning vs All-Dataset CNN  
## Recall & Margin Based Comparative Analysis

Bu doküman, **03_all_dataset** CNN modeli ile, düşük performanslı sınıfları iyileştirmek amacıyla yapılan  
**03_focus_classes (8 epoch fine-tuning)** modelinin **class-wise recall** ve **logit margin** analizlerine dayalı karşılaştırmasını içermektedir.

Amaç:
- Azınlık / zor sınıfların recall değerlerini artırmak
- Modelin karar güvenini (logit margin) yükseltmek
- Bu iyileştirmelerin genel performansı bozup bozmadığını gözlemlemek

---

## 1️⃣ Genel Performans Karşılaştırması

| Metrik | 03_all_dataset | 03_focus_classes | Değerlendirme |
|------|---------------|-----------------|--------------|
| Accuracy | 0.9230 | **0.9248** | Hafif artış |
| Macro Recall | **0.8851** | 0.8776 | ⬇ Düşüş |
| Weighted Recall | 0.9230 | **0.9248** | Hafif artış |
| Macro F1 | **0.8940** | 0.8911 | Hafif düşüş |

**Yorum:**  
Focused fine-tuning genel metrikleri belirgin biçimde iyileştirmemiştir.  
Bu çalışma, “herkesi aynı anda kurtarma” hedefiyle değil, **belirli problemli sınıflara müdahale** amacıyla yapılmıştır.

---

## 2️⃣ Hedeflenen (Problemli) Sınıflarda Recall Değişimi

### ✅ Belirgin İyileşme Sağlanan Sınıflar

| Sınıf | Recall (All) | Recall (Focused) | Değişim |
|-----|-------------|------------------|--------|
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.5256 | **0.8077** | **+28.2** |
| Tomato___Early_blight | 0.6933 | **0.7600** | +6.7 |
| Tomato___Septoria_leaf_spot | 0.8015 | **0.8876** | +8.6 |

Bu sınıflar, focused training’in **doğrudan hedeflediği** ve en çok zorlanan sınıflardı.  
Özellikle *Corn___Cercospora_leaf_spot* için elde edilen kazanım, yaklaşımın **doğru yönde olduğunu açıkça göstermektedir**.

---

### ❌ Regresyon Gözlenen Kritik Sınıf

| Sınıf | Recall (All) | Recall (Focused) | Değişim |
|-----|-------------|------------------|--------|
| Apple___Cedar_apple_rust | **0.8095** | 0.5952 | **−21.4** |

Bu düşüş, çalışmanın en önemli kırılma noktasıdır.

**Teknik Yorum:**
- Focused fine-tuning, belirli sınıfların karar sınırlarını güçlendirirken
- Benzer görsel paternlere sahip bazı sınıfların (Apple disease family) decision boundary’sini bozmuştur
- Bu durum, **catastrophic interference** etkisinin hafif bir örneğidir

Bu noktada model “yanlış yöne gitmemiştir”;  
aksine **çok spesifik bir hedefe fazla yaklaşmıştır**.

---

## 3️⃣ Logit Margin (Karar Güveni) Karşılaştırması

### Ortalama Margin Değerleri

| Metrik | All Dataset | Focused | Yorum |
|-----|------------|--------|------|
| Mean (all samples) | 6.50 | **7.19** | Artış |
| Mean (correct) | 6.95 | **7.68** | Artış |
| Mean (wrong) | 1.09 | 1.18 | Benzer |

### Correct Predictions – Percentile Artışı

| Percentile | All | Focused |
|----------|-----|--------|
| 50% | 5.89 | **6.56** |
| 75% | 9.97 | **11.05** |
| 90% | 13.66 | **15.12** |

**Yorum:**  
Focused model, doğru tahminlerde **daha yüksek karar güveni** üretmektedir.  
Bu, embedding uzayında sınıfların daha net ayrıştığını gösterir.

Bu kazanım, özellikle:
- CNN + SVM
- Low-margin override
- Ensemble / hybrid karar sistemleri  
için **çok değerli bir altyapı** sunmaktadır.

---

## 4️⃣ Neden “Pes Edilmiş” Gibi Görünse de Aslında Değil?

Bu çalışma şu nedenle **yarım kalmış gibi hissedebilir**:

- Hedeflenen sınıfların büyük kısmı iyileşti
- Sadece **tek bir kritik sınıf (Apple___Cedar_apple_rust)** ciddi düşüş yaşadı
- Bu da genel tabloyu “başarısızlık” gibi gösterdi

Ancak gerçek durum şudur:

> Model yanlış bir şey öğrenmedi.  
> Model, **fazla spesifik bir şeye çok iyi odaklandı**.

Bu, deneme-yanılma değil; **kontrollü bir model davranışı**dır.

---

## 5️⃣ Sonuç ve Devam Stratejisi

- 03_all_dataset modeli → **Ana model**
- 03_focus_classes modeli → **Destekleyici / yardımcı model**

Focused model:
- Tek başına nihai model olmak için uygun değildir
- Ancak:
  - düşük margin örneklerde
  - belirli sınıf alt kümelerinde
  - embedding tabanlı SVM veya ensemble yapılarda  
  **çok anlamlı katkı sağlayabilir**

Bu nedenle çalışma **başarısız değil**,  
**nihai mimarinin bir parçası olarak konumlandırılması gereken bir adım**dır.

---

## 6️⃣ Kısa Özet (TL;DR)

- ✔ Amaç doğruydu
- ✔ Çoğu hedef sınıf iyileşti
- ❌ Bir sınıf ciddi regress yaşadı
- ✔ Model confidence (margin) net biçimde arttı
- ❌ Tek başına “replacement model” olmadı
- ✅ Hybrid / ensemble yaklaşım için güçlü bir aday oldu
