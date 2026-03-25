# Neden Bu Projede PyTorch Kullandım? (Keras Yerine)

Bu doküman, **Bitki Hastalığı Sınıflandırma Projesi** kapsamında neden **TensorFlow / Keras yerine PyTorch** tercih edildiğini akademik ve mühendislik bakış açısıyla açıklamak için hazırlanmıştır.

---

## 1. Problem Tanımı ve Önceki Deneyim

Bu projeden önce yapılan çalışmada, aşağıdaki yaklaşım denenmiştir:

* Tek CNN modeli
* Tek softmax çıkışı
* Bitki türü + hastalık bilgisi **birleşik sınıf** olarak ele alınmıştır

  * Örnek: `Tomato___Early_blight`, `Potato___Late_blight`

Bu yaklaşımda model:

* Bazı sınıfları hiç öğrenememiş
* Internal prediction testlerinde bile

  * **X bitkisine ait görüntü → Y bitkisinin hastalığı** gibi
    mantıksal olarak hatalı sonuçlar üretmiştir

Bu durum, modelden yanlış bir soyutlama istenmesinden kaynaklanmıştır.

---

## 2. Keras Yaklaşımının Sınırlılıkları

Keras, yüksek seviyeli (high-level) bir framework olarak aşağıdaki özelliklere sahiptir:

* `model.fit()` ile eğitimin tek satırda yürütülmesi
* Forward pass, loss hesaplama ve backpropagation süreçlerinin
  kullanıcıdan gizlenmesi
* Otomatik callback yapıları (EarlyStopping vb.)

Bu yapı:

* Hızlı prototipleme için uygundur
* Ancak **eğitim sürecinin iç dinamiklerini görünmez kılar**

Önceki çalışmada bu durum:

* Overfitting nedenlerini tam analiz edememeye
* Modelin neden bazı sınıfları hiç öğrenemediğinin
  net olarak tespit edilememesine
  neden olmuştur.

---

## 3. PyTorch’un Tercih Edilme Gerekçeleri


### 3.1 Eğitim Sürecinin Açık Olması

PyTorch’ta:

* Forward pass
* Loss hesaplama
* Backpropagation
* Optimizer adımları

kullanıcı tarafından **açık şekilde yazılır**.

Bu sayede:

* Modelin hangi aşamada ne öğrendiği
* Hangi noktada overfitting başladığı
* Eğitim ve validation kayıplarının nasıl oluştuğu

bilinçli şekilde takip edilebilir.

---

### 3.2 Problem İzolasyonuna Uygunluk

Bu projede bilinçli olarak şu strateji benimsenmiştir:

1. Önce **tek bitki türü (domates)** için hastalık sınıflandırma CNN’i stabilize etmek
2. Daha sonra **ayrı bir bitki türü sınıflandırma modeli** eğitmek
3. Nihai sistemde **hiyerarşik / pipeline mimari** kullanmak

PyTorch, bu yaklaşım için daha uygundur çünkü:

* Çıkış boyutları ve model sorumlulukları net şekilde tanımlanır
* Tek softmax altında birleşik sınıflar oluşturma hatası kolayca engellenir

---

### 3.3 Akademik ve Araştırma Odaklılık

PyTorch:

* Akademik literatürde yaygın olarak kullanılan framework’tür
* CNN + SVM gibi hibrit yaklaşımlara doğal olarak uygundur
* Feature extractor olarak CNN kullanımı daha kontrollü şekilde yapılabilir

Bu proje kapsamında:

* CNN’in yalnızca sınıflandırıcı değil,
* Aynı zamanda **feature extractor** olarak değerlendirilmesi
  planlandığından PyTorch tercih edilmiştir.

---

### 3.4 Debug ve Hata Analizi Yeteneği

PyTorch ile:

* Batch bazlı analiz
* Sınıf bazlı hata incelemesi
* Confusion matrix üzerinden mantıksal hata takibi

çok daha net yapılabilmektedir.

Bu da geçmişte yaşanan:

> “Model çalışıyor gibi ama neden yanlış yaptığı bilinmiyor”

probleminin tekrar yaşanmamasını sağlar.

---

## 4. Keras Tamamen Yanlış mı?

Hayır.

Keras:

* Hızlı prototipleme
* Üretim ortamları
* Standart sınıflandırma problemleri

için uygundur.

Ancak bu proje:

* Öğrenme odaklı
* Analiz ve mimari kararların önemli olduğu
* Geçmişte yaşanmış bir tasarım hatasını bilinçli olarak düzeltmeyi amaçlayan

bir çalışma olduğu için PyTorch daha doğru bir tercih olduğunu düşünüyorum.

---

## 5. Sonuç

Bu projede PyTorch kullanılması:

* Bir framework tercihi değil,
* **Bilinçli bir problem çözme ve mimari kararın sonucudur**.

Amaç:

* CNN’in gerçekten ne öğrendiğini anlayabilmek
* Eski birleşik sınıf hatasını tekrar yaşamamak
* Hiyerarşik, genişletilebilir ve analiz edilebilir bir sistem kurmaktır.

Bu nedenle bu projede **Keras yerine PyTorch** tercih edilmiştir.
