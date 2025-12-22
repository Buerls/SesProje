import os
import torch
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "dataset_mini.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed_tensors")

# Klasör yoksa oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_and_save():
    df = pd.read_csv(CSV_PATH)
    new_data = []

    print("Veriler tensör formatına dönüştürülüyor...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['path']
        label = row['label']

        # Dosya adını ID olarak kullanalım
        file_name = os.path.basename(file_path).replace('.mp3', '.pt')
        save_path = os.path.join(OUTPUT_DIR, file_name)

        try:
            # --- İŞLEME (Eskiden Dataset içindeydi, şimdi burada) ---
            y, sr = librosa.load(file_path, sr=16000, mono=True)

            # Süre sabitleme (5 sn = 80000 sample)
            max_len = 16000 * 5
            if len(y) > max_len:
                y = y[:max_len]
            else:
                padding = max_len - len(y)
                y = np.pad(y, (0, padding), mode='constant')

            # Mel Spectrogram
            mels = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, hop_length=512)
            mels_db = librosa.power_to_db(mels, ref=np.max)

            # Normalizasyon
            mels_norm = (mels_db - mels_db.mean()) / (mels_db.std() + 1e-6)

            # Tensor yap ve kaydet
            tensor = torch.tensor(mels_norm, dtype=torch.float16)  # float16 yer kazandırır
            torch.save(tensor, save_path)

            # Yeni listeye ekle
            new_data.append({
                "path": save_path,  # Artık .pt dosyasının yolunu tutuyoruz
                "label": label
            })

        except Exception as e:
            print(f"Hata: {file_path} -> {e}")

    # Yeni CSV'yi kaydet
    new_csv_path = os.path.join(BASE_DIR, "data", "dataset_processed.csv")
    pd.DataFrame(new_data).to_csv(new_csv_path, index=False)
    print(f"\n✅ İşlem tamam! Yeni veri seti: {new_csv_path}")


if __name__ == "__main__":
    process_and_save()