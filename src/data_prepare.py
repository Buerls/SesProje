import os
import glob
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_COUNT = 4000
MIN_SECONDS = 2.0
MAX_SECONDS = 10.0

DATA_PATHS = {
    "tr": os.path.join(BASE_DIR, "data", "tr", "clips", "*.mp3"),
    "ko": os.path.join(BASE_DIR, "data", "ko", "clips", "*.mp3"),
    "sv": os.path.join(BASE_DIR, "data", "sv", "clips", "*.mp3")
}


def prepare_dataset():
    data_list = []

    print(f"Hedef: Her dilden rastgele {TARGET_COUNT} temiz dosya seçiliyor...\n")

    for lang, path_pattern in DATA_PATHS.items():
        files = glob.glob(path_pattern)

        # Dosyaları karıştır (Shuffle)
        np.random.shuffle(files)

        print(f"--- {lang.upper()} işleniyor (Toplam havuz: {len(files)}) ---")

        count = 0
        durations = []

        # İlerleme çubuğu ile dosyaları kontrol et
        for f in tqdm(files):
            if count >= TARGET_COUNT:
                break

            try:
                # Sadece süresine bakmak için hızlı yükleme
                duration = librosa.get_duration(path=f)

                # Süre filtresi
                if MIN_SECONDS <= duration <= MAX_SECONDS:
                    data_list.append({
                        "path": f,
                        "label": lang,
                        "duration": duration
                    })
                    durations.append(duration)
                    count += 1
            except Exception as e:
                continue  # Hatalı dosyayı atla

    # DataFrame oluştur
    df = pd.DataFrame(data_list)

    # CSV olarak kaydet
    csv_path = os.path.join(BASE_DIR, "data", "dataset_mini.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Veri seti hazırlandı ve kaydedildi: {csv_path}")
    print(f"Toplam Örnek Sayısı: {len(df)}")
    print(df["label"].value_counts())

    return df


def visualize_data(df):
    # Süre dağılımını çizelim
    plt.figure(figsize=(10, 5))

    for lang in df['label'].unique():
        subset = df[df['label'] == lang]
        plt.hist(subset['duration'], bins=20, alpha=0.5, label=lang)

    plt.title(f"Seçilen Verilerin Süre Dağılımı (Her dilden {TARGET_COUNT} adet)")
    plt.xlabel("Saniye")
    plt.ylabel("Adet")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    df = prepare_dataset()
    visualize_data(df)