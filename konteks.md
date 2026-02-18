Konteks Proyek: Prediksi Tren Klaim Asuransi Kesehatan Individu
1. Deskripsi Umum Proyek Proyek ini adalah sebuah kompetisi analisis data (Kaggle) yang bertujuan untuk memitigasi risiko finansial akibat lonjakan klaim asuransi kesehatan. Tujuannya adalah membangun model prediktif yang akurat untuk membantu perusahaan melakukan seleksi risiko dan menjaga agar harga premi tetap terjangkau bagi masyarakat.
2. Detail Dataset Dataset mencakup periode dari 1 Januari 2024 hingga 31 Juli 2025. Terdiri dari dua file utama:
• Data_Polis.csv (4.096 data): 
Dataset Description
Dataset terdiri dari 4096 data polis asuransi kesehatan dan 5781 data klaim asuransi kesehatan yang terjadi pada periode 1 Januari 2024 s/d 31 Juli 2025
Terdapat 5 informasi yang diberikan terkait data polis, yaitu: nomor polis, plancode, gender, tanggal lahir, tanggal efektif polis, dan domisili, dan terdapat 13 informasi yang diberikan terkait data klaim asuransi kesehatan, yaitu: claim ID, nomor polis, reimburse/cashless, inpatient/outpatient, ICD Diagnosis, ICD Description, status klaim, tanggal pembayaran, tanggal pasien masuk RS, tanggal pasien keluar RS, nominal klaim yang disetujui, nominal biaya RS yang terjadi, dan lokasi RS.
Berkas Data
Data_Klaim.csv – Data transaksi klaim asuransi kesehatan hingga periode 2025-07-31.
Data_Polis.csv – Data induk yang berisi informasi seluruh polis aktif.
sample_submission.csv – Berkas contoh format pengumpulan (submisi) hasil prediksi/analisa.
Deskripsi Kolom
1. Data_Klaim.csv
Daftar kolom yang tersedia pada data transaksi klaim:
Claim ID: Identifikasi unik untuk setiap klaim asuransi kesehatan yang diajukan oleh tertanggung.
Nomor Polis: Nomor identitas unik dari polis asuransi milik tertanggung.
Reimburse/Cashless: Metode penyelesaian klaim; baik melalui sistem ganti rugi (reimburse) maupun tanpa tunai (cashless).
Inpatient/Outpatient: Kategori perawatan; Rawat Inap (Inpatient) atau Rawat Jalan (Outpatient).
ICD Diagnosis: Kode klasifikasi penyakit berdasarkan International Statistical Classification of Diseases and Related Health Problems Revision. Analisa dapat menggunakan kode spesifik (contoh: H26.9) atau pengelompokan umum (contoh: H26).
ICD Description: Deskripsi medis atau penjelasan mengenai diagnosis berdasarkan kode ICD terkait.
Status Klaim: Status pemrosesan klaim (Paid = Klaim telah dibayarkan, Pending = Klaim dalam proses verifikasi).
Tanggal Pembayaran Klaim: Tanggal ketika dana klaim dicairkan kepada nasabah atau pihak RS.
Tanggal Pasien Masuk RS: Tanggal dimulainya perawatan medis di rumah sakit.
Tanggal Pasien Keluar RS: Tanggal selesainya perawatan medis (kepulangan) dari rumah sakit.
Nominal Klaim Yang Disetujui: Nilai nominal biaya kesehatan yang disetujui untuk dibayarkan oleh perusahaan asuransi.
Nominal Biaya RS Yang Terjadi: Total tagihan biaya rumah sakit yang diajukan oleh nasabah.
Lokasi RS: Letak geografis atau wilayah tempat rumah sakit berada.
2. Data_Polis.csv
Daftar kolom yang tersedia pada data profil polis nasabah:
Nomor Polis: Nomor identitas unik yang menghubungkan data polis dengan data klaim.
Plan Code: Kode produk yang menentukan cakupan wilayah pertanggungan:
M-001: Wilayah pertanggungan Seluruh Dunia (Worldwide).
M-002: Wilayah pertanggungan regional Asia.
M-003: Wilayah pertanggungan domestik Indonesia.
Gender: Jenis kelamin pemegang polis/tertanggung.
Tanggal Lahir: Tanggal lahir tertanggung (digunakan untuk kalkulasi usia).
Tanggal Efektif Polis: Tanggal awal mulai berlakunya proteksi asuransi kesehatan.
Domisili: Wilayah tempat tinggal atau alamat resmi tertanggung.
3. Masalah Bisnis yang Dihadapi Terdapat lonjakan klaim asuransi kesehatan individu sebesar 25,5% pada periode Januari–Juni 2025 dibandingkan dengan periode yang sama di tahun 2024. Kenaikan ini mengancam keterjangkauan harga premi, sehingga diperlukan analisis untuk memprediksi faktor-faktor yang memengaruhi nilai klaim.
4. Tugas Pemodelan (Forecasting) Agent harus melakukan peramalan (forecasting) untuk periode Agustus hingga Desember 2025 (5 bulan). Ada tiga variabel target yang harus diprediksi setiap bulannya:
1. Trend Frekuensi: Jumlah kejadian klaim.
2. Trend Severitas: Nilai rata-rata per klaim (Total Klaim ÷ Frekuensi).
3. Trend Total Claim: Akumulasi nominal klaim yang disetujui.
5. Struktur Output & Evaluasi
• Format Submisi: File .CSV berisi 15 baris (5 bulan x 3 variabel) dengan format kolom id seperti 2025-08_Claim_Frequency.
Peserta diwajibkan untuk mengirimkan file prediksi melalui kaggle dalam format .CSV dengan struktur sebagai berikut:

id, value
2025-08_Claim_Frequency,0
2025-08_Claim_Severity,0
2025-08_Total_Claim,0
2025-09_Claim_Frequency,0
...


• Metrik Penilaian: Menggunakan Mean Absolute Percentage Error (MAPE) untuk mengukur akurasi relatif terhadap nilai aktual.

6. Instruksi Tambahan untuk Agent
• Gunakan data historis untuk menemukan pola musiman atau tren kenaikan biaya.
• Analisis hubungan antara Plan Code atau Diagnosis ICD terhadap besarnya nominal klaim.
• Fokus pada minimalisasi nilai MAPE di ketiga komponen target (Frekuensi, Severitas, Total) secara seimbang.
