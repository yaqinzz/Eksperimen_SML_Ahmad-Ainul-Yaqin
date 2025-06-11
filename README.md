# Eksperimen Sistem Machine Learning - Ahmad Ainul Yaqin

## Pengaturan Environment

- Membuat environment baru

```
python -m venv env
```

- Mengaktifkan environment

```
env\Scripts\activate #windows
source env/bin/activate # linux/macos
```

- Mematikan environment

```
deactivate
```

- Menginstall installasi yang dibutuhkan dalam project

```
pip install -r requirements.txt
```

- Menghidupkan environment notebook

```
python -m ipykernel install --user --name=env --display-name "Python env"
```

## Struktur Project

- `Lung_Cancer_Raw/` - Direktori berisi data mentah

  - `cancer patient data sets.csv` - Dataset mentah untuk analisis

- `preprocessing/` - Direktori untuk proses preprocessing data
  - `automate_Ahmad-Ainul-Yaqin.py` - Script otomasi preprocessing
  - `Salinan_dari_Template_Eksperimen_MSML.ipynb` - Notebook template eksperimen
  - `Lung_Cancer_preprocessing/` - Hasil preprocessing
    - `feature_names.txt` - Daftar nama fitur
    - `scaler.joblib` - Model scaler untuk normalisasi data
    - `X_test_scaled.csv` - Data testing yang telah dinormalisasi
    - `X_train_scaled.csv` - Data training yang telah dinormalisasi
    - `y_test.csv` - Label data testing
    - `y_train.csv` - Label data training

## Menjalankan Eksperimen

- Menjalankan notebook untuk eksplorasi data

```
jupyter notebook preprocessing/Salinan_dari_Template_Eksperimen_MSML.ipynb
```

- Menjalankan mlflow untuk tracking eksperimen

```
mlflow ui
```

- Menjalankan script preprocessing

```
python preprocessing/automate_Ahmad-Ainul-Yaqin.py
```

## Catatan

Pastikan environment sudah aktif sebelum menjalankan perintah-perintah di atas.
