# 🎙️ Ses Tabanlı Dil Sınıflandırma: TCN Modeli

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)



Bu proje, ham ses sinyallerini kullanarak konuşmacının dilini (**Türkçe, Korece, İsveççe**) tespit eden derin öğrenme tabanlı bir **Konuşulan Dil Tanıma (LID)** sistemidir. Model mimarisi olarak, geleneksel RNN/LSTM yapılarının yerine daha hızlı ve kararlı olan **Zamansal Evrişimli Ağlar (TCN)** tercih edilmiştir.

---

## 🚀 Proje Özellikleri
- **Mimari:** Temporal Convolutional Network (TCN) - Dilated Convolutions.
- **Giriş Verisi:** Mel-Spectrogram (16kHz, 5 Saniye Sabit).
- **Başarı Oranı:** Test setinde **%89.67** Doğruluk.
- **Veri Zenginleştirme:** SpecAugment (Zaman ve Frekans Maskeleme).
- **Optimizasyon:** AdamW, Cosine Annealing Scheduler, Gradient Clipping.

---

## 📂 Proje Dizini

```text
SesProje/
├── data/                  # Veri setinin bulunduğu klasör (CSV ve .npy dosyaları)
├── models/                # Eğitilen model dosyaları (.pth)
│   └── best_model_pro.pth # %89.67 başarımı olan final model
├── src/                   # Kaynak kodlar
│   ├── dataset.py         # PyTorch veri yükleyici sınıfı
│   ├── model.py           # TCN model mimarisi
│   ├── train.py           # Eğitim döngüsü (Training Loop)
│   ├── preprocess_save.py # Sesleri spektrograma çevirip kaydetme
│
├── requirements.txt       # Gerekli kütüphaneler
└── README.md              # Proje dokümantasyonu
```

## 🛠️ Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

**1. Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install torch torchaudio librosa numpy pandas matplotlib seaborn scikit-learn tqdm
```

**2. Veri Setini Hazırlayın:**

preprocess_save.py dosyasını çalıştırarak ses dosyalarını işlenmiş .npy formatına çevirin. Bu işlem eğitimi 10 kat hızlandırır.
```bash
python src/preprocess_save.py
```

## 🧠 Model Eğitimi

Modeli sıfırdan eğitmek için aşağıdaki komutu kullanın:

```bash
python src/train.py
```

```text
Eğitim Parametreleri:

Epoch: 40

Batch Size: 64

Learning Rate: Cosine Annealing (Dalgalı)

Loss: CrossEntropy (Label Smoothing: 0.1)
```

## 📊 Sonuçlar ve Performans


| Metrik | Değer |
| :--- | :--- |
| **Eğitim Doğruluğu** | %87.50 |
| **Test Doğruluğu** | **%89.67** |
| **En İyi Epoch** | 27 |


## ⚠️ Önemli: Veri Seti Kurulumu

Projeyi çalıştırmadan önce **Mozilla Common Voice** veri setlerini indirip aşağıdaki klasör yapısına göre düzenlemeniz gerekmektedir.

1.  Proje ana dizininde `data` isminde bir klasör oluşturun.
2.  Bu klasörün içine `tr` (Türkçe), `ko` (Korece) ve `sv` (İsveççe) isimli 3 alt klasör açın.
3.  İndirdiğiniz veri setlerini (.mp3 dosyalarını içeren `clips` klasörü ve `.tsv` dosyalarını) ilgili dil klasörünün içine çıkarın.

**Olması Gereken Klasör Yapısı:**
```text
SesProje/
├── data/
│   ├── tr/          # Türkçe veri seti dosyaları buraya
│   │   ├── clips/
│   │   └── train.tsv
│   ├── ko/          # Korece veri seti dosyaları buraya
│   └── sv/          # İsveççe veri seti dosyaları buraya
├── src/
└── ...