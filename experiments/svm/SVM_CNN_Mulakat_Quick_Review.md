
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



