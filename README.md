# 📘 Judul Proyek
*Analisis dan Prediksi Konsumsi Energi Listrik Kota Tetouan Menggunakan Pendekatan Machine Learning dan Deep Learning*

## 👤 Informasi
- **Nama:** Richo Novian Saputra  
- **Repo:** https://github.com/richonovians/DataScience-ProjectUAS 
- **Video:** https://youtu.be/PwtCBIWj4sY   

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi konsumsi daya listrik (Power Consumption) di **Zone 1** Kota Tetouan menggunakan data deret waktu (*time series*).
- Menyelesaikan permasalahan prediksi beban listrik (*load forecasting*) untuk efisiensi energi di Kota Tetouan.
- Melakukan *data preparation* mencakup *cleaning*, *chronological splitting*, dan *feature engineering* (terutama Lag Features).
- Membangun dan membandingkan 3 model: **Linear Regression (Baseline)**, **Random Forest (Advanced)**, dan **Deep Learning (MLP)**.
- Melakukan evaluasi komprehensif menggunakan RMSE, MAE, dan R² Score untuk menentukan pendekatan terbaik.

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Operator jaringan listrik menghadapi tantangan dalam menyeimbangkan pasokan dan permintaan energi secara *real-time*.
- Ketidakakuratan prediksi dapat menyebabkan inefisiensi operasional, pemborosan energi, atau risiko kegagalan sistem (*blackout*).

**Goals:**  
- Membangun model prediksi konsumsi listrik (khususnya Zone 1) dengan target akurasi tinggi ($R^2 > 0.90$).
- Menganalisis perbandingan performa antara model linear sederhana dengan model *Deep Learning* yang kompleks.
- Mengidentifikasi variabel historis dan cuaca yang paling mempengaruhi pola konsumsi.

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset & Hasil Evaluasi
│   ├── tetouan_power_raw.csv
│   └── model_comparison_results.csv # Tabel hasil evaluasi
│
├── notebooks/              # Jupyter Notebook utama
│   └── DataScience_ProyekMachineLearning.ipynb
│
├── src/                    # Source code .py
│   └── datascience_proyekmachinelearning.py
│   
├── models/                 # Model Artifacts
│   ├── model_baseline.pkl           # Model 1: Linear Regression
│   ├── model_rf.pkl                 # Model 2: Random Forest
│   └── model_dl_final.h5            # Model 3: Deep Learning
│
├── images/                 # Output Visualisasi
│   ├── comparison_error_metrics.png
│   ├── comparison_r2_score.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_target_distribution.png
│   ├── eval_prediction_comparison.png
│   ├── model_dl_training_history.png
│   ├── model_lr_actual_vs_pred.png
│   └── model_rf_feature_importance.png
│
├── requirements.txt        # Dependencies
├── .gitignore
├── Checklist Submit.md
├── LICENSE
├── Laporan Proyek Machine Learning.pdf
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository (Tetouan City Power Consumption).
- **Jumlah Data:** 52.000 baris (Data time-series per 10 menit).
- **Tipe:** Time Series Regression.

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
| `DateTime` | Timestamp data. |
| `Temperature` | Suhu rata-rata (°C). |
| `Humidity` | Kelembaban (%). |
| `Wind Speed` | Kecepatan angin (m/s). |
| `Zone 1 Power` | **Target** (Konsumsi Listrik KW). |
| `lag_1` | Konsumsi listrik 1 jam yang lalu (Feature Engineering). |
| `lag_24` | Konsumsi listrik jam yang sama kemarin (Feature Engineering). |

---

# 4. 🔧 Data Preparation
- **Cleaning:** Menghapus *missing values* (NaN) yang terbentuk akibat proses *lagging*.
- **Transformasi:**
  - *Feature Engineering:* Membuat fitur waktu (Hour, Month, DayOfWeek) dan fitur historis (Lag & Rolling Mean).
  - *Scaling:* Standard Scaling (Z-score).
- **Splitting:** Menggunakan *Chronological Split* (80% Train, 20% Test) untuk mencegah kebocoran data masa depan (*data leakage*).

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** **Linear Regression**. Model sederhana untuk menangkap hubungan linear kuat dari fitur *lag*.
- **Model 2 – Advanced ML:** **Random Forest Regressor**. Model *ensemble* dengan *Hyperparameter Tuning* (`n_estimators`, `max_depth`) menggunakan RandomizedSearchCV.
- **Model 3 – Deep Learning:** **Multi-Layer Perceptron (MLP)**. Arsitektur Neural Network (256-128-64 neuron) dengan Dropout dan Learning Rate Scheduler.

---

# 6. 🧪 Evaluation
**Metrik:** RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), dan R² Score.

### Hasil Singkat
| Model | Score (R²) | Catatan |
|-------|--------|---------|
| **Baseline (LR)** | **0.995** | **Model Terbaik**. Sangat cepat dan akurat. |
| Advanced (RF) | 0.981 | Sedikit *overfitting* / sulit menangkap tren linear halus. |
| Deep Learning | 0.994 | Performa sangat kompetitif, mendekati baseline. |

*(Nilai RMSE terendah juga diraih oleh Baseline: 457 KW)*

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** **Linear Regression (Baseline)**.
- **Alasan:** Fitur historis (`lag_1`) memiliki korelasi linear yang sangat kuat dengan target. Model sederhana mampu menangkap pola ini secara efektif tanpa kompleksitas berlebih.
- **Insight penting:** Dalam prediksi jangka pendek (*short-term forecasting*), rekayasa fitur (*feature engineering*) seringkali lebih berdampak signifikan daripada kompleksitas algoritma model.

---

# 8. 🔮 Future Work
- [x] Tambah data (Hari libur nasional/Event khusus).
- [x] Tuning model (Coba algoritma XGBoost/LightGBM).
- [x] Coba arsitektur DL lain (LSTM/GRU untuk *sequence modeling*).
- [x] Deployment (Buat API dengan FastAPI/Streamlit).

---

# 9. 🔁 Reproducibility
Gunakan environment dengan menjalankan `pip install -r requirements.txt`. Berikut versi utama yang digunakan:

```text
ucimlrepo==0.0.7
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
matplotlib==3.10.0
seaborn==0.13.2
tensorflow==2.19.0
joblib==1.5.2
xgboost==3.1.2
