# CNN-Based Plant Disease Classification  
## Fine-Tuning ve Targeted Augmentation Denemeleri – Teknik Analiz

Bu doküman, **38 sınıflı (bitki + hastalık birleşik)** CNN modelinde  
yaşanan sınıf-bazlı recall problemlerini çözmek için yapılan iki ana müdahaleyi ve  
bu müdahalelerin neden başarısız olduğunu **teknik olarak** açıklamaktadır.

Amaç yalnızca “sonuç olmadı” demek değil;  
**neden olmadığını** ölçümlerle ve mimari düzeyde ortaya koymaktır.

---

## 1️⃣ Başlangıç Noktası: Baseline CNN (03 All Dataset)

### Model Özeti
- Tek CNN (from scratch)
- 38 sınıf (bitki + hastalık birleşik)
- Accuracy ≈ 0.923
- Macro Recall ≈ 0.885
- Global olarak stabil

### Tespit Edilen Problem
Bazı sınıflarda belirgin recall düşüşleri vardı:

- `Corn___Cercospora_leaf_spot` (~0.53)
- `Tomato___Early_blight` (~0.69)
- `Potato___healthy` (~0.62)

Bu sınıflar:
- Ya görsel olarak çok benzer hastalıklara sahipti
- Ya da “healthy” gibi bağlama duyarlı sınıflardı

---

## 2️⃣ Deneme #1 – Class Rebalancing + Fine-Tuning

### ❓ Mülakat Sorusu:  
**“Class rebalancing’i nasıl yaptın?”**

### 🔧 Uygulanan Teknik
- `WeightedRandomSampler` kullanıldı
- Sadece problemli sınıflar oversample edildi
- Diğer sınıflar doğal dağılımda bırakıldı
- Öğrenme oranı düşürüldü (fine-tuning)
- Model önceden eğitilmiş ağırlıklarla başlatıldı

Teknik olarak:
- Aynı batch içinde problemli sınıfların görülme olasılığı artırıldı
- Loss fonksiyonu **global** kaldı (CrossEntropy)
- Optimizer tüm ağırlıkları güncelledi

### ❌ Sonuç
- Hedeflenen sınıflarda recall **arttı**
- Ancak:
  - Daha önce güçlü olan sınıflarda recall **düştü**
  - Global denge bozuldu

### 🔬 Teknik Açıklama (neden bozuldu?)
- Sampler ile dağılım değiştirildi
- Ancak loss ve optimizer **tüm modeli** güncelledi
- Decision boundary **global olarak kaydı**
- Bu, lokal iyileştirme yerine **global trade-off** yarattı

> Yani problemli sınıflar “daha önemli” hale geldi  
> ama bu önem, diğer sınıfların temsilini zayıflattı.

---

## 3️⃣ Deneme #2 – Targeted Augmentation (6 Epoch)

Fine-tuning sonrası görülen bozulma nedeniyle,
daha kontrollü bir yaklaşım denendi.

### ❓ Mülakat Sorusu:  
**“Targeted augmentation’i nasıl yaptın?”**

### 🔧 Uygulanan Teknik
- `WeightedRandomSampler` **kullanılmadı**
- Dataset wrapper ile:
  - Problemli sınıflara **daha agresif augmentation**
  - Diğer sınıflara **standart transform**
- Backbone donduruldu
- Sadece classifier head güncellendi
- 6 epoch kısa eğitim

Amaç:
- Global dağılımı bozmadan
- Sadece intra-class variance artırmak

### 📊 Gözlenen Sonuçlar

#### 🔴 Recall
Hedeflenen sınıflar **beklenenin aksine çöktü**:

| Sınıf | Baseline | Targeted Aug |
|------|---------|--------------|
| Corn___Cercospora | ~0.53 | ~0.06 |
| Tomato___Early_blight | ~0.69 | ~0.27 |
| Potato___healthy | ~0.62 | 0.00 |

Ayrıca bazı sağlam sınıflar da zarar gördü:
- Apple rust
- Peach healthy
- Raspberry healthy

#### 🔴 Margin Analizi
- Correct prediction margin ↑
- **Wrong prediction margin da ↑**
- Yanlış tahminlerde model daha **emin** hale geldi

Bu, klasik bir **overconfidence** sinyalidir.

---

## 4️⃣ Neden İki Yöntem de Başarısız Oldu?

Bu noktada kritik farkındalık şudur:

> Problem **veri azlığı değil**,  
> problem **temsil çakışmasıdır**.

Tek CNN aynı anda:
1. Bitki türünü ayırt etmeye
2. Hastalık türünü ayırt etmeye

zorlanmaktadır.

Özellikle:
- Aynı hastalık adı farklı bitkilerde
- “Healthy” gibi bağlama bağımlı sınıflar

CNN’in feature space’inde **çatışma yaratmaktadır**.

Ne rebalancing ne augmentation:
- Bu çatışmayı çözemez
- Sadece karar sınırlarını oynatır

---

## 5️⃣ Sonuç: Mimari Karar (Kanıta Dayalı)

Bu deneylerden sonra şu sonuca varılmıştır:

> Tek CNN üzerinde yapılan lokal müdahaleler  
> (rebalancing, fine-tuning, targeted augmentation)  
> problemi çözmemekte,  
> aksine global genelleme dengesini bozmaktadır.

Bu nedenle:
- Problem tanımı ayrılmalıdır

### Önerilen Mimari
- **Model-1:** Bitki türü sınıflandırma
- **Model-2:** Bitkiye özel hastalık sınıflandırma

Bu karar:
- Deneysel
- Ölçüme dayalı
- Margin + recall analizleriyle desteklidir

---

## 6️⃣ Mülakat İçin Net Cümle (Özet)

> “Tek CNN üzerinde class rebalancing ve targeted augmentation denedim.  
> Rebalancing global decision boundary’yi kaydırdı,  
> targeted augmentation ise overconfidence yarattı.  
> Margin analizi ve class-wise recall sonuçları,  
> problemin veri değil mimari kaynaklı olduğunu gösterdi.  
> Bu yüzden problem tanımını ayırmaya karar verdim.”

