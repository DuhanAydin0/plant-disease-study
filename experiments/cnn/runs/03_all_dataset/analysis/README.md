# CNN 03 – Sınıf Odaklı Müdahale ve SVM ile Birleştirme Kararı

## 1. Mevcut Durumun Özeti

CNN 03 modeli, tüm veri seti üzerinde eğitildiğinde:

- Test Accuracy: ~0.92
- Macro Recall: ~0.88
- Weighted Recall: ~0.92

Genel performans yüksek olmasına rağmen, **bazı sınıflarda recall değerlerinin belirgin şekilde düşük kaldığı** gözlemlenmiştir. Bu durum, modelin genel doğruluğu yüksek olsa bile **küçük veya görsel olarak benzer sınıflarda örnekleri kaçırdığını** göstermektedir.

---

## 2. Sınıf Bazlı (Class-wise) Analiz Bulguları

Class-wise recall analizi sonucunda özellikle şu sınıflar dikkat çekmiştir:

- Corn___Cercospora_leaf_spot (recall ≈ 0.53)
- Tomato___Early_blight (recall ≈ 0.69)
- Potato___healthy (recall ≈ 0.62)
- Apple___Apple_scab / Black_rot gibi bazı hastalık sınıfları

Bu sınıfların ortak özellikleri:
- Görece daha az örnek sayısı
- Görsel olarak başka hastalıklarla yüksek benzerlik
- CNN’in bu sınıflar için kararsız tahminler üretmesi

Bu noktada problem **overfitting değil**, aksine **decision boundary’nin bu sınıflar için yeterince optimal olmaması** olarak yorumlanmıştır.

---

## 3. Logit Margin Analizi ve Yorumu

Logit margin analizi şu önemli sonuçları vermiştir:

- Mean margin (correct predictions): ~6.95
- Mean margin (wrong predictions): ~1.09

Bu bulgular şu anlama gelmektedir:

- Doğru tahminlerin büyük kısmı **yüksek güvenle** yapılmaktadır.
- Yanlış tahminler ise çoğunlukla **kararsız (low-confidence) bölgede** yer almaktadır.
- Model yanlış tahminleri “emin olarak” yapmamaktadır.

Bu sonuçlar, **CNN’in feature space’te sınıfları ayırabildiğini**, ancak bazı sınıflar için **karar sınırının yeterince rafine olmadığını** göstermektedir.

---

## 4. Neden Genel Augmentation Değil?

04_all_dataset_augmentation deneyinde:

- Train accuracy ciddi şekilde düşmüş
- Test performansı 03 modeline kıyasla gerilemiştir

Bu durum, global augmentation’ın:
- Öğrenilmiş güçlü temsilleri bozduğunu
- Modeli gereksiz zorladığını
- Hedef sınıflar yerine tüm veri dağılımını etkilediğini göstermiştir

Bu nedenle **genel augmentation yaklaşımı terk edilmiştir**.

---

## 5. Alınan Karar: Sınıf Odaklı Müdahale + CNN + SVM

Yukarıdaki analizler doğrultusunda şu strateji benimsenmiştir:

### Aşama 1 – Sınıf Odaklı CNN Müdahalesi
- Sadece recall’ı düşük sınıflara odaklanılacak
- Hafif ve kontrollü augmentasyonlar uygulanacak
  - Düşük açılı rotation
  - Sınırlı zoom
  - Aşırı geometrik veya renk bozucu dönüşümlerden kaçınılacak
- Amaç: feature space’i bozmak değil, **zayıf sınıfların temsillerini güçlendirmek**

### Aşama 2 – CNN + SVM Hibrit Yapı
- CNN, **feature extractor** olarak kullanılacak
- Son katman (logit öncesi) feature’lar alınacak
- Bu feature’lar üzerinde SVM eğitilecek
- Amaç: 
  - CNN’in öğrendiği güçlü temsilleri korumak
  - Kararsız bölgelerde decision boundary’yi iyileştirmek

---

## 6. Teknik ve Kavramsal Sonuç

Bu kararın arkasındaki temel gerekçe şudur:

> CNN modeli kapasite sınırında değildir.  
> Sınıflar feature space’te ayrılabilir durumdadır.  
> Ancak bazı sınıflarda karar sınırı suboptimaldir.  
> Bu nedenle CNN + SVM birleşimi, problemi mimariyi büyütmeden çözmek için en mantıklı adımdır.

Bu yaklaşım hem:
- Teknik olarak gerekçelidir
- Hem de mülakatlarda savunulabilir bir modelleme kararıdır

---

## 7. Sonraki Adımlar

- Sınıf-odaklı fine-tuning / augmentasyon denemeleri
- CNN feature extractor çıktılarının kaydedilmesi
- Linear ve RBF SVM karşılaştırması
- Recall odaklı iyileşmenin ölçülmesi




===== Class-wise Recall Analysis (03 Model) =====

                                               precision    recall  f1-score   support

                           Apple___Apple_scab     0.8152    0.7895    0.8021        95
                            Apple___Black_rot     0.9620    0.8085    0.8786        94
                     Apple___Cedar_apple_rust     1.0000    0.8095    0.8947        42
                              Apple___healthy     0.8626    0.9113    0.8863       248
                          Blueberry___healthy     0.9651    0.9779    0.9714       226
                      Cherry___Powdery_mildew     0.9613    0.9371    0.9490       159
                             Cherry___healthy     0.9104    0.9457    0.9278       129
   Corn___Cercospora_leaf_spot Gray_leaf_spot     0.7455    0.5256    0.6165        78
                           Corn___Common_rust     0.9887    0.9722    0.9804       180
                  Corn___Northern_Leaf_Blight     0.7647    0.8725    0.8150       149
                               Corn___healthy     0.9405    0.9943    0.9667       175
                            Grape___Black_rot     0.8663    0.9153    0.8901       177
                 Grape___Esca_(Black_Measles)     0.9531    0.8798    0.9150       208
   Grape___Leaf_blight_(Isariopsis_Leaf_Spot)     0.9688    0.9568    0.9627       162
                              Grape___healthy     0.9508    0.9062    0.9280        64
     Orange___Haunglongbing_(Citrus_greening)     0.9916    1.0000    0.9958       827
                       Peach___Bacterial_spot     0.9259    0.9393    0.9326       346
                              Peach___healthy     0.9412    0.8727    0.9057        55
                Pepper,_bell___Bacterial_spot     0.9098    0.8013    0.8521       151
                       Pepper,_bell___healthy     0.8898    0.9417    0.9150       223
                        Potato___Early_blight     0.9728    0.9533    0.9630       150
                         Potato___Late_blight     0.9104    0.8133    0.8592       150
                             Potato___healthy     0.8333    0.6250    0.7143        24
                          Raspberry___healthy     0.9444    0.8947    0.9189        57
                            Soybean___healthy     0.9636    0.9712    0.9674       764
                      Squash___Powdery_mildew     0.9603    0.9638    0.9620       276
                     Strawberry___Leaf_scorch     0.9805    0.9042    0.9408       167
                         Strawberry___healthy     0.9559    0.9420    0.9489        69
                      Tomato___Bacterial_spot     0.9064    0.9688    0.9366       320
                        Tomato___Early_blight     0.6980    0.6933    0.6957       150
                         Tomato___Late_blight     0.7351    0.8606    0.7929       287
                           Tomato___Leaf_Mold     0.8667    0.9028    0.8844       144
                  Tomato___Septoria_leaf_spot     0.9106    0.8015    0.8526       267
Tomato___Spider_mites Two-spotted_spider_mite     0.8577    0.9087    0.8825       252
                         Tomato___Target_Spot     0.8522    0.8160    0.8337       212
       Tomato___Tomato_Yellow_Leaf_Curl_Virus     0.9875    0.9789    0.9832       805
                 Tomato___Tomato_mosaic_virus     0.8361    0.8947    0.8644        57
                             Tomato___healthy     0.9874    0.9833    0.9854       240

                                     accuracy                         0.9230      8179
                                    macro avg     0.9072    0.8851    0.8940      8179
                                 weighted avg     0.9245    0.9230    0.9227      8179



===== Logit Margin Analysis (03 Model) =====
Mean margin (all samples): 6.4955
Mean margin (correct predictions): 6.9466
Mean margin (wrong predictions): 1.0895

Margin percentiles (correct predictions):
10% percentile: 1.6511
25% percentile: 3.2249
50% percentile: 5.8930
75% percentile: 9.9672
90% percentile: 13.6560