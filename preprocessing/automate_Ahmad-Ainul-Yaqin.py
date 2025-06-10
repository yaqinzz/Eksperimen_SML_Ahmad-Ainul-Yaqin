import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def praproses_data(ukuran_pengujian=0.2, kondisi_acak=42):
    """
    Fungsi untuk melakukan praproses data kanker paru-paru.
    
    Parameter:
    ukuran_pengujian (float): Porsi data yang digunakan untuk pengujian, nilai default 0.2 (20%)
    kondisi_acak (int): Nilai seed untuk memastikan reproduksibilitas hasil, nilai default 42
    
    Return:
    tuple: Data yang telah melalui proses praproses (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print("Memulai proses praproses data...")
    print("Sedang membaca berkas data.csv...")

    try:
        direktori_dasar = os.path.dirname(os.path.abspath(__file__))

        jalur_data = os.path.join(direktori_dasar, "..", "Lung_Cancer_Raw", "cancer patient data sets.csv")
        jalur_data = os.path.abspath(jalur_data) 

        df = pd.read_csv(jalur_data)
        print(f"Berkas berhasil dibaca dari lokasi: {jalur_data}")
    except FileNotFoundError:
        print(f"Kesalahan: Berkas cancer patient data sets.csv tidak ditemukan pada jalur yang diharapkan: {jalur_data}")
        print("Pastikan berkas 'cancer patient data sets.csv' tersedia di dalam folder 'Lung_Cancer_Raw' (satu tingkat di atas folder 'preprocessing').")
        return None, None, None, None, None
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca berkas cancer patient data sets.csv: {e}")
        return None, None, None, None, None

    df_terproses = df.copy()

    if 'Level' in df_terproses.columns and df_terproses['Level'].dtype == 'object':
        print("Melakukan pengkodean label pada kolom 'Level'...")
        df_terproses['Level'] = df_terproses['Level'].map({'High': 2, 'Medium': 1, 'Low': 0})
    elif 'Level' not in df_terproses.columns:
        print("Kesalahan: Kolom 'Level' tidak ditemukan dalam kumpulan data.")
        return None, None, None, None, None

    print("Menghapus kolom 'Patient Id' dan 'index' yang tidak diperlukan...")
    kolom_yang_dihapus = []
    if 'Patient Id' in df_terproses.columns:
        kolom_yang_dihapus.append('Patient Id')
    if 'index' in df_terproses.columns:
        kolom_yang_dihapus.append('index')

    if kolom_yang_dihapus:
        df_terproses = df_terproses.drop(columns=kolom_yang_dihapus, errors='ignore')

    print("Memisahkan fitur (X) dan target (y) untuk proses pemodelan...")
    try:
        X = df_terproses.drop(columns=['Level'])
        y = df_terproses['Level']
    except KeyError:
        print("Kesalahan: Gagal memisahkan fitur dan target. Pastikan kolom 'Level' tersedia setelah praproses.")
        return None, None, None, None, None


    print("Melakukan pembagian data pelatihan dan pengujian...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ukuran_pengujian, random_state=kondisi_acak, stratify=y
    )

    print("Melakukan standardisasi fitur untuk meningkatkan performa model...")
    penskalaan = StandardScaler()
    X_train_terskalakan = penskalaan.fit_transform(X_train)
    X_test_terskalakan = penskalaan.transform(X_test)

    nama_folder_keluaran = "Lung_Cancer_preprocessing"
    direktori_praproses = os.path.join(direktori_dasar, nama_folder_keluaran)

    os.makedirs(direktori_praproses, exist_ok=True)

    X_train_terskalakan_df = pd.DataFrame(X_train_terskalakan, columns=X.columns)
    X_test_terskalakan_df = pd.DataFrame(X_test_terskalakan, columns=X.columns)

    print(f"Menyimpan data yang telah melalui praproses ke folder: {direktori_praproses}")
    X_train_terskalakan_df.to_csv(os.path.join(direktori_praproses, "X_train_scaled.csv"), index=False)
    X_test_terskalakan_df.to_csv(os.path.join(direktori_praproses, "X_test_scaled.csv"), index=False)

    y_train_df = y_train.to_frame(name='diagnosis')
    y_test_df = y_test.to_frame(name='diagnosis')
    y_train_df.to_csv(os.path.join(direktori_praproses, "y_train.csv"), index=False)
    y_test_df.to_csv(os.path.join(direktori_praproses, "y_test.csv"), index=False)

    jalur_penskalaan = os.path.join(direktori_praproses, "scaler.joblib")
    joblib.dump(penskalaan, jalur_penskalaan)
    print(f"Penskalaan berhasil disimpan di lokasi: {jalur_penskalaan}")

    if isinstance(X, pd.DataFrame):
        jalur_nama_fitur = os.path.join(direktori_praproses, "feature_names.txt")
        with open(jalur_nama_fitur, 'w') as f:
            for nama_fitur in X.columns:
                f.write(f"{nama_fitur}\n")
        print(f"Daftar nama fitur berhasil disimpan di lokasi: {jalur_nama_fitur}")

    print("Seluruh data yang telah melalui praproses berhasil disimpan!")

    return X_train_terskalakan, X_test_terskalakan, y_train, y_test, penskalaan

if __name__ == "__main__":
    print("Program praproses data kanker paru-paru dimulai...")
    hasil = praproses_data()
    if hasil is not None and hasil[0] is not None:
        print("Proses praproses data telah selesai dengan sukses.")
    else:
        print("Proses praproses data gagal atau tidak menghasilkan keluaran yang diharapkan.")
    print("Program telah selesai dijalankan!")
