# LAPORAN PROYEK MACHINE LEARNING - NAJMAH FEMALEA
## - Project Overview
**Latar Belakang**<br>
Dalam era digital saat ini, industri fashion telah mengalami transformasi besar melalui e-commerce. Pengguna kini memiliki akses ke ribuan pilihan produk fashion melalui platform daring, seperti marketplace dan toko online. Namun, banyaknya opsi sering kali membingungkan pelanggan, yang dapat menyebabkan kesulitan dalam memilih produk yang sesuai dengan preferensi, kebutuhan, atau gaya mereka. Di sinilah sistem rekomendasi memainkan peran penting. Sistem rekomendasi bertujuan untuk menyaring informasi dan memberikan saran yang relevan kepada pelanggan berdasarkan data, seperti riwayat pembelian, preferensi, atau perilaku pengguna. Dengan menerapkan teknologi rekomendasi, perusahaan dapat meningkatkan pengalaman pelanggan, memperkuat loyalitas, dan mendorong penjualan. [1]

**Pentingnya Proyek**<br>
Proyek ini bertujuan untuk mengembangkan sistem rekomendasi fashion berbasis data yang dapat membantu pelanggan menemukan produk yang sesuai dengan preferensi mereka. Pentingnya proyek ini terletak pada:

- Meningkatkan Kepuasan Pelanggan: Sistem rekomendasi yang personal dapat memberikan saran yang relevan, menghemat waktu pelanggan dalam mencari produk.
- Meningkatkan Penjualan: Rekomendasi yang relevan terbukti dapat meningkatkan nilai transaksi dan jumlah pembelian.
- Efisiensi Pengelolaan Produk: Memberikan wawasan kepada perusahaan tentang tren fashion dan perilaku pelanggan, sehingga dapat mengoptimalkan stok dan kampanye pemasaran.

## -Business Understanding 
**Problem Statements**
1. Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
2. Dengan data rating yang dimiliki, bagaimana perusahaan dapat merekomendasikan produk yang mungkin disukai dan belum pernah dikunjungi oleh pengguna?

**Goals**
1. Menghasilkan sejumlah rekomendasi produk yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
2. Menghasilkan sejumlah rekomendasi produk yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik collaborative filtering.

**Solution statements** <br>
Untuk menjawab problem statements dan memenuhi goals diatas, maka rekomendasi ini akan menggunakan dua algoritma sistem rekomendasi yaitu *content based filtering* dan *collaborative filtering*.

## - Data Understanding
1. Data yang digunakan berasal dari kaggel bisa diakses dilink [berikut](https://www.kaggle.com/datasets/bhanupratapbiswas/fashion-products). Dimana dataset ini memiliki **1000 baris** dan **9 kolom**.
2. Fitur dalam Dataset: 
    1. User ID : merupakan ID dari pengguna
    2. Product ID : merupakan ID dari produk
    3. Product Name : merupakan nama dari produk, dimana terdapat 5 jenis produk yaitu Dress, T-Shirt, Jeans, Sweater dan Shoes.
    4. Brand : merupakan merk atau brand dari produk, dimana terdapat 5 tipe brand yaitu Adidas, H&M, Zara, Gucci, dan Nike.
    5. Category : merupakan kategori dari produk, dimana terdapat 3 kategori yaitu anak-anak, wanita dan pria.
    6. Price : merupakan harga dari produk
    7. rating : merupakan rating yang didapatkan dari produk
    8. color : merupakan warna dari produk, dimana terdapat 6 jenis warna yaitu Black, Yellow, White, Blue, Green, dan Red.
    9. size : merupakan ukuran dari produk, dimana terdapat 4 tipe size yaitu XL, L, S, dan M.

3. Diketahui dengan menggunakan fungsi **info()** bahwa: <br>
   a. kolom **user ID, product ID, dan price** bertipe data *integer*<br>
   b. kolom **rating** bertipe data *float*,<br>
   c. kolom **product name, brand, category, color, size** bertipe data *object*.<br>

4. Missing Value<br>
Dengan menggunakan fungsi **isnull().sum()** pada library pandas diketahui bahwa tidak terdapat nilai yang hilang pada data ini.

5. Duplicate Data<br>
Dengan menggunakan fungsi **duplicated().sum()** pada library pandas diketahui bahwa tidak terdapat nilai yang duplikasi pada data ini.

## - Data Preparation
Pada tahap ini penting dilakukan sebelum memulai pemodelan yang digunakan untuk memformat ulang atau merestrukturisasi data, dan akhirnya menggabungkan data untuk dianalisis. 

1. Normalisasi pada Fitur Rating<br>
Karena fitur **Rating** bertipe data float yang memiliki banyak desimal, ini bisa dianggap sebagai noise dalam beberapa konteks. Maka, bulatkan dengan 1 angka dibelakang koma untuk mengurangi gangguan ini. Sehingga dari data yang bernilai 1.043159 menjadi 1.0.
2. Ubah Fitur<br>
Ada beberapa hal yang diubah yaitu :<br>
    - Ubah Nama Fitur <br>
    Karena nama fitur yang tidak memiliki format yang tepat, maka ubah nama fitur agar tidak ada spasi dan huruf kecil semua, seperti kolom 'Product ID' menjadi 'product_id' dan seterusnya.
    - Ubah nilai yang ada di fitur product dan user<br>
    Karena pada nilai tersebut hanya berisikan angka agar lebih mudah dibaca dan dipahami, mengubah nilai dari Product ID dan User ID agar memiliki prefix product_ atau user_ di depan setiap nilainya.
3. Feature Engineering
- Content Based Filtering hanya menggunakan fitur **product_name, category, price, brand, color dan size**, karena Metode ini fokus pada features (atribut) dari produk untuk merekomendasikan produk yang mirip dengan produk yang pernah diinteraksi oleh pengguna. Representasi atribut dilakukan dengan metode TF-IDF pada product_name dan One-Hot Encoding untuk atribut kategorikal seperti category dan brand. <br>

## - Pemodelan
**1. Content Based Filtering**
- Kelebihan:
  - Personalisasi yang tinggi:
    Content Based Filtering memberikan rekomendasi yang disesuaikan dengan preferensi individu berdasarkan atribut atau fitur dari item yang disukai pengguna.
  - Tidak memerlukan data pengguna lain:
    Content Based Filtering hanya membutuhkan informasi tentang pengguna dan item, sehingga cocok untuk sistem dengan sedikit pengguna (tidak bergantung pada komunitas).
  - Dapat menjelaskan rekomendasi:
    Karena berbasis atribut item, sistem dapat menjelaskan mengapa suatu item direkomendasikan (misalnya, "Produk ini mirip dengan produk yang Anda sukai sebelumnya").

- Kekurangan:
  - Masalah overspecialization:
    Sistem hanya merekomendasikan item yang mirip dengan item yang sudah disukai pengguna, sehingga sulit menemukan item yang berbeda atau mengeksplorasi pilihan baru.
  - Ketergantungan pada fitur item:
    Jika atribut atau fitur item tidak lengkap atau kurang representatif, kualitas rekomendasi akan menurun.

- Training
    - Mengambil data<br>
    Mengambil 5 sampel secara acak menggunakan fungsi sample() menggunakan enam fitur yang sudah ditentukan tadi, sebagai berikut:
      ![image](https://github.com/user-attachments/assets/f750727c-1e3e-451c-9bac-b2abc35bb821)
    
    - TF-TDF Vectorizer<br>
    Menggunakan TfidfVectorizer dari library scikit-learn untuk memproses teks menjadi representasi numerik (vektor TF-IDF) dari data product_name yang menghasilkan matriks dengan 1000 baris dan 5 kolom. Lalu mengubah matriks jarang (sparse matrix) menjadi matriks penuh (dense matrix) dan disimpan di dalam dataframe. 
    
    - Cosine Silimarity<br>
    Pendekatan Content-Based Filtering menggunakan informasi atribut produk untuk memberikan rekomendasi berdasarkan kemiripan antar produk. Model ini bekerja dengan menggunakan Cosine Similarity sebagai metrik untuk mengukur kesamaan antar produk yang direpresentasikan dalam matriks kesamaan (similarity matrix).
        - Matriks Similarity: Matriks ini sebelumnya telah dihitung menggunakan TF-IDF (untuk atribut teks) atau metode representasi lain untuk atribut produk. Matriks berisi nilai kesamaan antara setiap pasangan produk, dengan nilai antara 0 hingga 1. Semakin tinggi nilai cosine similarity, semakin mirip dua produk tersebut.
     
    - Mendapatkan Rekomendasi<br>
        - Fungsi product_recommendations menggunakan metode argpartition untuk menemukan indeks dari produk-produk dengan nilai kesamaan tertinggi terhadap produk yang dimasukkan sebagai input. Hal ini dilakukan dengan mempartisi data berdasarkan nilai tertinggi dalam urutan tertentu.
        - Pembuangan Produk Input: Produk yang menjadi input (product_id) dikeluarkan dari daftar rekomendasi untuk memastikan hanya produk lain yang muncul dalam hasil.
        - Penggabungan Data: Setelah produk-produk yang mirip ditemukan, produk tersebut digabungkan kembali dengan data asli menggunakan pd.merge, sehingga informasi lengkap (seperti product_name atau atribut lainnya) dapat disertakan dalam hasil.
    
    - Parameter yang Digunakan
        - product_id: Merupakan ID produk yang menjadi referensi untuk mencari rekomendasi. Contoh: 'product_265'.
        - similarity_data: Matriks kesamaan (Cosine Similarity) antara produk, yang disiapkan sebelumnya.
        - items: Data asli yang memuat informasi lengkap tentang produk, seperti product_name, category, brand, dan lainnya.
        - k: Jumlah produk rekomendasi yang ingin ditampilkan. Dalam kode ini, menampilkan 10 kategori.
    
    - Hasil Rekomendasi:
    Memberikan 10 produk dengan nilai kesamaan tertinggi terhadap 'product_265' <br>
    ![image](https://github.com/user-attachments/assets/beb1ad3b-14c0-42a8-9884-8a75f0f51a0d)

**2. Collaborative Filtering**
- Kelebihan:
  - Kemampuan mengeksplorasi:
    Collaborative Filtering dapat merekomendasikan item yang tidak pernah diakses atau dilihat oleh pengguna tetapi populer di kalangan pengguna dengan preferensi serupa.
  - Tidak memerlukan data fitur item:
    Collaborative Filtering hanya membutuhkan interaksi pengguna-item (seperti rating, pembelian, atau klik) sehingga cocok untuk data yang kaya akan interaksi.
- Kekurangan:
  - Cold-start problem untuk pengguna baru:
    Collaborative Filtering tidak dapat memberikan rekomendasi kepada pengguna baru yang belum memiliki riwayat interaksi.
  - Ketergantungan pada data komunitas:
    Jika komunitas pengguna kecil atau datanya sparse (terlalu sedikit interaksi), kualitas rekomendasi akan menurun.
  - Masalah skalabilitas:
    Ketika jumlah pengguna atau item bertambah sangat besar, proses pencocokan (similarity computation) menjadi mahal secara komputasi.

- Training
    - Feature Engineering<br>
    Collaborative Filtering menggunakan fitur **user_id, rating, product_name, dan product_id**, karena Metode ini fokus pada pola interaksi antara pengguna dan produk untuk membuat rekomendasi, berdasarkan data eksplisit seperti Rating atau data historis pembelian. Lalu ntuk mempermudah pemrosesan oleh model TensorFlow, dilakukan encoding pada kolom user_id dan product_id.
    - Pendekatan Collaborative Filtering menggunakan model deep learning berbasis TensorFlow untuk mempelajari hubungan antara pengguna dan produk. Model ini memanfaatkan embeddings untuk merepresentasikan pengguna dan produk dalam ruang vektor:
        1. Data Encoding:
            - ID pengguna (user_id) dan produk (product_id) diubah menjadi format numerik melalui proses encoding, menghasilkan peta antara ID asli ke nilai numerik.
            - Peta ini digunakan untuk memastikan data input model memiliki format numerik yang sesuai.
        2. Normalisasi Rating:
        Rating produk dinormalisasi ke dalam skala 0 hingga 1 untuk memastikan model bekerja secara konsisten.
         3. Split Data:<br>
            - Mengacak Dataset<br>
              Mengacak dataset (shuffling data) agar data tidak memiliki urutan yang berpotensi memengaruhi hasil analisis atau pelatihan model.
            - Split data<br>
              Data dibagi menjadi training set dan validation set dengan rasio 80:20 menggunakan train_test_split untuk melatih dan mengevaluasi model. Dimana: <br>
              - fitur X nya adalah customer dan product
              - fitur y nya adalah rating.
    - Training model<br>
        1. Arsitektur Model RecommenderNet:<br>
        - Model terdiri dari:
            - User Embedding Layer: Mempelajari representasi pengguna dalam ruang vektor.
            - Product Embedding Layer: Mempelajari representasi produk dalam ruang vektor.
            - User Bias dan Product Bias: Menangkap bias unik dari pengguna atau produk tertentu.
        - Vektor embedding pengguna dan produk di-dot product untuk menghasilkan skor kesamaan. Skor ini ditambahkan dengan bias pengguna dan produk, lalu diaktifkan menggunakan fungsi sigmoid untuk menghasilkan prediksi rating.
        - Model dilatih menggunakan:
          - Loss function: Binary Crossentropy untuk membandingkan prediksi dengan nilai sebenarnya.
          - Optimizer: Adam dengan learning rate 0.001.
          - Metrics: Root Mean Squared Error (RMSE) untuk mengukur performa model.
          - Epoch: 100, model akan melakukan iterasi 100 kali.
        - Visualisasi Metrik:
        Grafik pelatihan pada nilai training dan validation loss dapat dilihat pada gambar berikut:
        ![image](https://github.com/user-attachments/assets/ba034279-918a-48c3-84eb-88ab8e9fa5ce)
    
        2. Prediksi Rekomendasi:
            - Produk yang belum pernah dibeli oleh pengguna dievaluasi menggunakan model.
            - Model menghasilkan prediksi rating untuk produk tersebut, dan produk dengan prediksi tertinggi direkomendasikan kepada pengguna
            - Hasil Top 10 Rekomendasi:<br>
        Menampilkan produk yang Sudah Dibeli Pengguna yaitu produk dengan
        rating tertinggi yang sebelumnya dibeli oleh pengguna.<br> 
        ![image](https://github.com/user-attachments/assets/bacfd5ba-f630-4ef7-b8f2-acf5cc7741c5)<br>
        Menampilkan Rekomendasi Produk Baru: Produk yang belum pernah dibeli oleh pengguna tetapi memiliki prediksi rating tertinggi. <br>
        ![image](https://github.com/user-attachments/assets/72a50174-5527-4dde-8fd5-f72aea45612d)

## - Evaluasi
1. Based Content Filtering
   - Presisi
    - Identifikasi True Positives (TP) dan False Positives (FP):
        - True Positives (TP): Jumlah rekomendasi yang relevan, yaitu produk yang benar-benar mirip dengan produk asal (product_265) berdasarkan atribut seperti brand, category, color, atau size.
        - False Positives (FP): Jumlah rekomendasi yang tidak relevan, yaitu produk yang tidak cukup mirip dengan produk asal berdasarkan kriteria yang sama.
    - Kriteria Penilaian:
    Dalam kasus ini, relevansi dapat ditentukan berdasarkan atribut seperti:
        - Brand: Jika brand produk sama dengan produk asal (Gucci).
        - Category: Jika kategori produk sama dengan produk asal (Kids' Fashion).
        - Color: Jika warna produk sama dengan produk asal (White).
        - Size: Jika ukuran produk sama dengan produk asal (XL)
    - Perhitungan Presisi:<br>
      Rumus Presisi:<br>
      ![image](https://github.com/user-attachments/assets/d3173f1f-673c-4879-9100-4847e3f7e571)
    - Hasil Evaluasi:
        - True Positives (TP): 7 produk dianggap relevan karena memenuhi salah satu atau lebih kriteria (brand, category, color, atau size yang mirip dengan produk asal product_265).
        - False Positives (FP): 3 produk tidak relevan karena tidak memenuhi kriteria.
        - Presisi:<br>
        ![image](https://github.com/user-attachments/assets/385eca1e-99b0-4099-991c-762bdfc682db)<br>
    Sistem rekomendasi ini memiliki presisi sebesar 70%, yang berarti 70% dari produk yang direkomendasikan relevan dengan produk asal (product_265). Hasil ini menunjukkan bahwa mayoritas rekomendasi cukup sesuai dengan preferensi pengguna, tetapi masih ada ruang untuk perbaikan

2. Collaborative Filtering
- Root Mean Squared Error
Evaluasi dilakukan untuk memahami performa model berdasarkan metrik yang digunakan serta memastikan solusi memenuhi business understanding dan problem statement. 
    - Metrik Evaluasi:
        - Root Mean Squared Error (RMSE):
        Metrik ini mengukur rata-rata akar dari kuadrat selisih antara rating prediksi dan rating aktual.
        RMSE yang lebih rendah menunjukkan model lebih akurat dalam memprediksi rating.
        Formula: <br>
        ![image](https://github.com/user-attachments/assets/08e6cb95-e6bf-455d-bb0e-f56229960420)

        - Loss Function (Binary Crossentropy):
        Mengukur perbedaan antara probabilitas prediksi dan nilai aktual, cocok untuk masalah klasifikasi seperti prediksi apakah pengguna akan menyukai produk.
        - Hasil Evaluasi:
        Pada akhir pelatihan model:
            - RMSE pada Training Set: 0.1920
            - RMSE pada Validation Set: 0.3806<br>
        Hasil ini menunjukkan bahwa model memiliki kemampuan yang baik untuk memprediksi rating pengguna terhadap produk. Perbedaan kecil antara training loss dan validation loss mengindikasikan model tidak overfitting. RMSE berada pada tingkat yang rendah, artinya model dapat memberikan rekomendasi yang cukup akurat.

## - Kesimpulan
Dengan tujuan untuk memberikan rekomendasi produk yang relevan kepada pengguna berdasarkan pola pembelian sebelumnya. Model berhasil memberikan rekomendasi produk yang belum pernah dibeli pengguna, dengan mempertimbangkan kesamaan produk dan preferensi pengguna. Model juga berhasil mencapai goals tersebut dengan menyarankan produk baru yang memiliki potensi tinggi untuk diminati pengguna berdasarkan pola interaksi. Model Collaborative Filtering ini mampu memenuhi kebutuhan bisnis dalam memberikan rekomendasi produk yang relevan dan personal kepada pengguna. Dengan evaluasi menggunakan RMSE, hasil menunjukkan bahwa model dapat memprediksi rating dengan tingkat akurasi yang memadai, mendukung pencapaian tujuan bisnis dan memberikan dampak positif terhadap pengguna maupun perusahaan.

**Dampak Positif:**
- Untuk Bisnis: Rekomendasi yang personal dapat meningkatkan peluang pembelian, memperpanjang customer lifetime value, dan meningkatkan pendapatan.
- Untuk Pengguna: Pengguna mendapatkan pengalaman belanja yang lebih efisien dan relevan, mengurangi waktu pencarian produk.


## - Referensi
[1] Alkaff, M., Khatimi, H. and Eriadi, A. (2020). Sistem Rekomendasi Buku pada Perpustakaan Daerah Provinsi Kalimantan Selatan Menggunakan Metode Content-Based Filtering. MATRIK : Jurnal Manajemen, Teknik Informatika dan Rekayasa Komputer, 20(1), pp.193–202. doi:https://doi.org/10.30812/matrik.v20i1.617.
