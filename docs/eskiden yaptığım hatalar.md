# Hatalı İlk Girişim Analizim (Plant Disease Classification)

## Arka Plan

Bu projeyi ilk olarak,öğrenciyken, CNN ve klasik makine öğrenmesi yöntemleriyle henüz yeterli pratik tecrübem yokken
denedim. Amacım bitki yapraklarından hastalık sınıflandırması yapmaktı.  
Ancak veri hazırlama, problem tanımı ve modelleme aşamalarında bazı temel tasarım hataları yaptım.

Bu doküman, o ilk denememde yaptığım hataların ve sonrasında edindiğim teknik çıkarımların
bilinçli bir analizidir.

---

## Tespit Ettiğim Temel Hatalar

### 1. Çift Augmentation (Double Augmentation)

- Zaten **offline olarak augment edilmiş** bir veri seti kullandım.
- Buna ek olarak `ImageDataGenerator` ile **tekrar augmentation** uyguladım.
- Bu durum görsellerin aşırı bozulmasına ve hastalıkla ilgili ayırt edici görsel sinyallerin kaybolmasına yol açtı.

**Sonuç:**  
Modelin gerçek veri dağılımını öğrenmesini engelledim; overfitting ve genelleme problemi oluştu.

---

### 2. Train ve Validation Veri Dağılımı Uyumsuzluğu

- Eğitim verisini augmented dataset’ten aldım.
- Validation verisini non-augmented dataset’ten aldım.
- Bu durum train ve validation setleri arasında **distribution mismatch** oluşturdu.

**Sonuç:**  
Validation skorları güvenilir olmaktan çıktı ve model performansını yanlış yorumladım.

---

### 3. Validation Split Mantığını Yanlış Kullanmam

- `validation_split` parametresini kullandım.
- Ancak train ve validation verilerini **farklı klasörlerden** çektim.
- Bu yapı `validation_split` mantığıyla uyumlu değildi.

**Sonuç:**  
Validation mekanizması teknik olarak çalışıyor gibi görünse de, istatistiksel olarak hatalıydı.

---

### 5. Görüntü Verisi için Uygunsuz Oversampling

- Zaten augmented olan görüntüler üzerinde `RandomOverSampler` kullandım.
- Bu yöntem görüntü verisinde aynı örneklerin birebir kopyalanmasına yol açtı.

**Sonuç:**  
SVM modelim aynı örnekleri tekrar tekrar görerek yanıltıcı şekilde yüksek performans gösterdi.

---

### 6. Yüksek Boyutlu Feature Space ile Aşırı Grid Search

- 128x128 boyutlu görüntüleri flatten ederek SVM’e verdim.
- Çok sayıda kernel ve parametre kombinasyonunu GridSearch ile denedim.
- Bu yapı hem hesaplama açısından verimsizdi hem de gürültüye aşırı duyarlıydı.

---

## Çıkardığım Teknik Dersler

- Veri stratejisinin model karmaşıklığından daha kritik olduğunu gördüm.
- Augmentation’ın yalnızca **train-time** ve **tek aşamada** uygulanması gerektiğini öğrendim.
- Train ve validation setlerinin aynı dağılımdan gelmesi gerektiğini anladım.
- Yüksek doğruluğun her zaman doğru öğrenme anlamına gelmediğini fark ettim.
