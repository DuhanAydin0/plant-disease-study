
# SVM (01) ve SVM (02) Sonuçları – Bulgular ve Teknik Gözlemler

Bu doküman, **Basic SVM (01)** ve **Optimize Edilmiş SVM (02)** modellerinin
sonuçlarını karşılaştırmalı olarak değerlendirmek ve bu sonuçlardan çıkarılan
teknik bulguları **parça parça, okunabilir ve profesyonel** şekilde kayıt altına almak amacıyla hazırlanmıştır.
Bu içerik, proje sonunda oluşturulacak ana README.md dosyasına kaynak teşkil edecektir.

---

##  Basic SVM (01) vs Optimized SVM (02) – Sayısal Karşılaştırma

###  Train Accuracy
- **01 (basic):** 0.989  
- **02 (optimized):** 0.958  

 Beklenen durum gerçekleşmiştir.  
Regularization ve `class_weight` kullanımı sayesinde modelin **ezberleme eğilimi azalmıştır**.  
Bu, teorik olarak **olumlu ve istenen** bir etkidir.

---

###  Validation Accuracy
- **01 (basic):** 0.8687  
- **02 (optimized):** 0.8514  

❌ Validation accuracy düşmüştür.  
Bu durum kritik olup, optimizasyonun **genelleme performansını artırmadığını** göstermektedir.

---

## Sınıf Bazlı Kritik İnceleme (Asıl Önemli Kısım)

###  Early Blight (En problemli sınıf)

| Model | Precision | Recall |
|------|-----------|--------|
| 01   | 0.64      | 0.53   |
| 02   | 0.55      | 0.59   |

 Recall değeri bir miktar artmıştır.  
 Ancak precision ciddi şekilde düşmüştür.  
 Sonuç olarak genel doğruluk zarar görmüştür.



---

###  Tomato Mosaic Virus

| Model | Recall |
|------|--------|
| 01   | 0.71   |
| 02   | 0.76   |

Bu sınıfta küçük bir iyileşme gözlemlenmiştir.  
Ancak:
- Sınıfın örnek sayısı çok düşüktür.
- Overall performansı anlamlı şekilde etkilememektedir.

---

### Tomato Yellow Leaf Curl Virus (Baskın sınıf)

- Recall değeri **0.96 → 0.90** düşmüştür. 

 `class_weight` kullanımının bedeli bu sınıfta ödenmiştir.  
 Baskın sınıf, azınlık sınıflar lehine **fedakârlık yapmıştır**.

---

##  Macro Avg vs Weighted Avg (Öğretici Karşılaştırma)

###  01 (Basic)
- **Macro F1:** ≈ 0.83  
- **Weighted F1:** ≈ 0.87  

###  02 (Optimized)
- **Macro F1:** ≈ 0.82  
- **Weighted F1:** ≈ 0.85  

 Macro F1 iyileşmemiştir.  
 Weighted F1 gerilemiştir.

 Bu durum şunu göstermektedir:  
**Optimizasyon sınıf dengesini teorik olarak iyileştirmiştir ancak genelleme kapasitesini artırmamıştır.**

---

##4 En Önemli Teknik Çıkarım

Bu sonuçlar net biçimde şunu **kanıtlamaktadır**:

> **SVM’nin bu problemdeki temel sorunu hiperparametre değil, temsil (feature) problemidir.**

- C düşürülmüştür → ezberleme azalmıştır  
- `class_weight` eklenmiştir → sınıfsal adalet artmıştır  
- Ancak sınıflar hâlâ karışmaktadır

Sebep:
- Flatten edilmiş piksel vektörleri
- Uzamsal (spatial) bilginin kaybolması
- Doku, leke ve şekil örüntülerinin öğrenilememesi

---

##  Bu Noktada Verilmesi Gereken Doğru Karar

 SVM süreci **akademik olarak kapatılmıştır**.  
Ancak bu kapanış **güçlü ve gerekçelidir**.


