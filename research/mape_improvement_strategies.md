# Strategi Meningkatkan MAPE dengan TimesFM

Berdasarkan dokumentasi Google Research TimesFM dan praktik terbaik _forecasting_, berikut adalah strategi untuk meningkatkan akurasi (menurunkan MAPE):

## 1. Gunakan Prediksi Median (q50)

Secara statistik, menggunakan **Median** (nilai tengah) meminimalkan _Mean Absolute Error_ (dan MAPE), sedangkan **Mean** (rata-rata) meminimalkan _Mean Squared Error_ (MSE).

- **Mengapa:** Data klaim asuransi sering memiliki distribusi yang miring (_skewed_) atau outliers. Mean akan tertarik oleh outliers, sedangkan Median lebih robust.
- **Implementasi:** Ambil output quantile ke-5 (q50) dari `quantile_forecast`.

## 2. Fitur `continuous_quantile_head`

TimesFM 2.5 memiliki opsi `use_continuous_quantile_head=True`.

- **Mengapa:** Ini memungkinkan model menghasilkan distribusi probabilitas yang lebih halus dan akurat untuk estimasi ketidakpastian, yang berujung pada estimasi titik tengah (median) yang lebih baik.

## 3. Normalisasi Input

Pastikan `normalize_inputs=True` aktif.

- **Mengapa:** TimesFM dilatih dengan berbagai skala data. Normalisasi lokal per-jendela waktu (context window) membantu model fokus pada pola relatif (tren/musiman) daripada besaran absolut yang mungkin berbeda dari data latihannya.

## 4. Covariates (Advanced - Belum diterapkan)

Menambahkan variabel eksternal seperti Hari Libur Nasional atau Indikator Ekonomi.

- **Mengapa:** Klaim kesehatan mungkin memiliki pola musiman terkait liburan atau wabah.
- **Catatan:** Memerlukan setup yang lebih kompleks (install `jax` + data masa depan untuk covariates).

## 5. Log Transformation (Opsional)

Melakukan transformasi `Log(x+1)` pada target sebelum prediksi.

- **Mengapa:** Mengubah distribusi data yang eksponensial menjadi lebih normal, memudahkan model untuk belajar.

---

**Tindakan:**
Script `predict_timesfm.py` akan diperbarui untuk mengimplementasikan **Poin 1, 2, dan 3** secara langsung.
