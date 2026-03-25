
# SVM Sonuçları Sonrası – Tam Öğrenme ve Mülakat Notları

Bu doküman, **SVM sonuçlarının paylaşılmasından itibaren** yapılan tüm teknik açıklamaları,
yorumları, karar gerekçelerini ve mülakat için gerekli mantıkları **eksiksiz ve bütünlüklü**
şekilde kayıt altına almak amacıyla oluşturulmuştur.

---

## 1. SVM Train Sonuçlarının Yorumu

### Genel Durum
- Train Accuracy ≈ %99
- Confusion matrix neredeyse tamamen diyagonal

Bu durum şunu gösterir:
> Model train verisini çok iyi öğrenmiş, hatta büyük ölçüde ezberlemiştir.

Bu tek başına iyi bir sonuç değildir; **genelleme yeteneği** ölçülmeden karar verilemez.

---

## 2. Validation Sonuçlarının Yorumu

- Validation Accuracy ≈ %87
- Train–Validation farkı ≈ %12

Bu fark:
- Net bir **overfitting** göstergesidir.
- Problemin yalnızca hiperparametre ayarlarıyla çözülemeyeceğini işaret eder.

Özellikle:
- *Early Blight*
- *Tomato Mosaic Virus*

gibi sınıflarda recall düşüktür. Bunun nedeni sınıfların **görsel olarak birbirine benzemesi**dir.

---

## 3. SVM Bu Problemde Neden Zorlanıyor?

SVM:
- Sabit (handcrafted veya flatten edilmiş) feature’larla çalışır
- Uzamsal (spatial) bilgi öğrenemez
- Doku, şekil ve lokal örüntüleri modelleyemez

Bu nedenle:
> Görüntü tabanlı, çok sınıflı ve görsel benzerliği yüksek problemlerde yapısal olarak sınırlıdır.

Bu bir hata değil, **model–problem uyumsuzluğudur**.

---

## 4. SVM Optimizasyonu Nedir?

SVM optimizasyonu şu parametreler üzerinden yapılır:

### 4.1 C (Regularization)
- Büyük C → agresif öğrenme → overfitting
- Küçük C → daha yumuşak sınır → daha iyi genelleme (bazen underfitting)

Amaç:
> Train–validation farkını azaltmak

---

### 4.2 Kernel Seçimi
- `linear`: basit, genelde yetersiz
- `rbf`: non-linear, en yaygın ve güçlü
- `poly`: riskli ve pahalı

Görüntü problemlerinde çoğunlukla **RBF** tercih edilir.

---

### 4.3 Gamma (RBF için)
- Büyük gamma → çok lokal karar sınırı → overfitting
- Küçük gamma → daha genel sınır → underfitting

---

### 4.4 class_weight
- Sınıf dengesizliğini azaltır
- Azınlık sınıfların recall’ını artırabilir
- Accuracy düşebilir

Bu:
> Adalet sağlar ama yeni bilgi öğretmez.

---

## 5. Neden SVM’yi Aşırı Optimize Etmedik?

- GridSearch / Cross-Validation pahalıdır
- Kazanım sınırlıdır (%1–2 civarı)
- Asıl problem hiperparametre değil, **feature öğrenememektir**

Bu nedenle:
> Minimal, öğretici optimizasyon yeterlidir.

Amaç:
- “Denendi ve sınırına ulaşıldı”yı kanıtlamak

---

## 6. SVM’nin Projedeki Rolü

SVM bu projede:
- Nihai model değil
- **Baseline (karşılaştırma modeli)** olarak kullanılmıştır


---

## 7. CNN’e Geçiş Kararının Mantığı

CNN:
- Feature’ları kendisi öğrenir
- Uzamsal ilişkileri korur
- Görsel benzerliği daha iyi ayırt eder

Bu yüzden:
> SVM’den CNN’e geçiş bir kaçış değil, mühendislik kararıdır.

---

# SVM → CNN Geçişi |

## Problem
- Çok sınıflı (10 sınıf) bitki hastalığı sınıflandırma
- Görsel olarak benzer hastalıklar (Early vs Late blight)
- Sınıf dengesizliği mevcut

---

## SVM Sonuçları (Özet)
- **Train Accuracy:** ~%99
- **Validation Accuracy:** ~%87
- **Yorum:** Belirgin overfitting, genelleme kaybı

Azınlık sınıflarda (Early blight, Mosaic virus) **recall düşük**.

---

## SVM Neden Zorlandı?
- SVM **sabit/flatten edilmiş feature** kullanır
- **Uzamsal (spatial) bilgi öğrenemez**
- Görüntüdeki doku/şekil ilişkilerini yakalayamaz

> Sorun hiperparametre değil, **model–problem uyumsuzluğu**.

---

## SVM Nasıl Optimize Edilebilirdi?
- **C:** Overfitting’i azaltmak için düşürülebilir
- **Kernel:** RBF tercih edilir
- **Gamma:** Daha genel sınır için düşürülebilir
- **class_weight:** Azınlık sınıfların recall’ını artırır
- **Scaling:** Zorunlu (StandardScaler)

> Optimizasyon **yumuşatma sağlar**, yeni bilgi öğretmez.

---

## Neden Full GridSearch Yapılmadı?
- Hesaplama maliyeti yüksek
- Kazanım sınırlı (%1–2)
- Asıl sınırlayıcı faktör: **feature öğrenememek**





