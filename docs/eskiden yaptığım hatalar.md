# Hatalı İlk Girişim Analizi (Plant Disease Classification)

## Arka Plan

Bu proje ilk olarak, CNN ve klasik makine öğrenmesi yöntemleriyle henüz yeterli pratik tecrübem yokken
denenmiştir. Amaç bitki yapraklarından hastalık sınıflandırması yapmaktı.  
Ancak veri hazırlama, problem tanımı ve modelleme aşamalarında bazı temel tasarım hataları yapılmıştır.

Bu doküman, o ilk denemede yapılan hataların ve sonrasında edinilen teknik çıkarımların
bilinçli bir analizidir.

---

## Tespit Edilen Temel Hatalar

### 1. Çift Augmentation (Double Augmentation)

- Zaten **offline olarak augment edilmiş** bir veri seti kullanıldı.
- Buna ek olarak `ImageDataGenerator` ile **tekrar augmentation** uygulandı.
- Bu durum görsellerin aşırı bozulmasına ve hastalıkla ilgili ayırt edici görsel sinyallerin kaybolmasına yol açtı.

**Sonuç:**  
Model gerçek veri dağılımını öğrenemedi, overfitting ve genelleme problemi oluştu.

---

### 2. Train ve Validation Veri Dağılımı Uyumsuzluğu

- Eğitim verisi augmented dataset’ten alındı.
- Validation verisi non-augmented dataset’ten alındı.
- Bu durum train ve validation setleri arasında **distribution mismatch** oluşturdu.

**Sonuç:**  
Validation skorları güvenilir olmaktan çıktı ve model performansı yanlış yorumlandı.

---

### 3. Validation Split Mantığının Yanlış Kullanımı

- `validation_split` parametresi kullanıldı.
- Ancak train ve validation verileri **farklı klasörlerden** çekildi.
- Bu yapı `validation_split` mantığıyla uyumlu değildir.

**Sonuç:**  
Validation mekanizması teknik olarak çalışıyor gibi görünse de, istatistiksel olarak hatalıydı.



### 5. Görüntü Verisi için Uygunsuz Oversampling

- Zaten augmented olan görüntüler üzerinde `RandomOverSampler` kullanıldı.
- Bu yöntem görüntü verisinde aynı örneklerin birebir kopyalanmasına yol açtı.

**Sonuç:**  
SVM modeli aynı örnekleri tekrar tekrar görerek yanıltıcı şekilde yüksek performans gösterdi.

---

### 6. Yüksek Boyutlu Feature Space ile Aşırı Grid Search

- 128x128 boyutlu görüntüler flatten edilerek SVM’e verildi.
- Çok sayıda kernel ve parametre kombinasyonu GridSearch ile denendi.
- Bu yapı hesaplama açısından verimsizdi.

---

## Çıkarılan Teknik Dersler

- Veri stratejisi, model karmaşıklığından daha kritiktir.
- Augmentation yalnızca **train-time** ve **tek aşamada** uygulanmalıdır.
- Train ve validation setleri aynı dağılımdan gelmelidir.
- Yüksek doğruluk her zaman doğru öğrenme anlamına gelmez.

---

