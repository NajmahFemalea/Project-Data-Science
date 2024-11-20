# LAPORAN PROYEK MACHINE LEARNING - NAJMAH FEMALEA
## - Project Overview
**Latar Belakang**
Dalam era digital saat ini, industri fashion telah mengalami transformasi besar melalui e-commerce. Pengguna kini memiliki akses ke ribuan pilihan produk fashion melalui platform daring, seperti marketplace dan toko online. Namun, banyaknya opsi sering kali membingungkan pelanggan, yang dapat menyebabkan kesulitan dalam memilih produk yang sesuai dengan preferensi, kebutuhan, atau gaya mereka. Di sinilah sistem rekomendasi memainkan peran penting.

Sistem rekomendasi bertujuan untuk menyaring informasi dan memberikan saran yang relevan kepada pelanggan berdasarkan data, seperti riwayat pembelian, preferensi, atau perilaku pengguna. Dengan menerapkan teknologi rekomendasi, perusahaan dapat meningkatkan pengalaman pelanggan, memperkuat loyalitas, dan mendorong penjualan. 

**Pentingnya Proyek**
Proyek ini bertujuan untuk mengembangkan sistem rekomendasi fashion berbasis data yang dapat membantu pelanggan menemukan produk yang sesuai dengan preferensi mereka. Pentingnya proyek ini terletak pada:

- Meningkatkan Kepuasan Pelanggan: Sistem rekomendasi yang personal dapat memberikan saran yang relevan, menghemat waktu pelanggan dalam mencari produk.
- Meningkatkan Penjualan: Rekomendasi yang relevan terbukti dapat meningkatkan nilai transaksi dan jumlah pembelian.
- Efisiensi Pengelolaan Produk: Memberikan wawasan kepada perusahaan tentang tren fashion dan perilaku pelanggan, sehingga dapat mengoptimalkan stok dan kampanye pemasaran.

## -Business Understanding 
**Problem Statements**
1. Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
2. Dengan data rating yang dimiliki, bagaimana perusahaan dapat merekomendasikan produk yang mungkin disukai dan belum pernah dikunjungi oleh pengguna?

**Goals**
1. Menghasilkan sejumlah rekomendasi restoran yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
2. Menghasilkan sejumlah rekomendasi restoran yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik collaborative filtering.

**Solution statements**
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

3. Diketahui dengan menggunakan fungsi **info()** bahwa :
  a. kolom **user ID, product ID, dan price** bertipe data *integer*
  b. kolom **rating** bertipe data *float*, 
  c. kolom **product name, brand, category, color, size** bertipe data *object*.

## - Data Preparation
Pada tahap ini penting dilakukan sebelum memulai pemodelan yang digunakan untuk membersihkan data kotor, memformat ulang atau merestrukturisasi data, dan akhirnya menggabungkan data untuk dianalisis. 
1. Missing Value
Dengan menggunakan fungsi **isnull().sum()** pada library pandas diketahui bahwa tidak terdapat nilai yang hilang pada data ini.
2. Duplicate Data
Dengan menggunakan fungsi **duplicated().sum()** pada library pandas diketahui bahwa tidak terdapat nilai yang duplikasi pada data ini.
3. Normalisasi
Karena fitur **Rating** bertipe data float yang memiliki banyak desimal, ini bisa dianggap sebagai noise dalam beberapa konteks. Maka, saya membulatkannya dengan 1 angka dibelakang koma untuk mengurangi gangguan ini. Sehingga dari data yang bernilai 1.043159 menjadi 1.0.
4. Ubah Nama Fitur
Karena nama fitur yang tidak memiliki format yang tepat, maka saya ubah nama fitur agar tidak ada spasi dan huruf kecil semua, seperti kolom 'Product ID' menjadi 'product_id' dan seterusnya.  
5. Memilih fitur yang akan dijadikan untuk pemodelan
Lalu saya melakukan feature engineering dimana untuk pemodelan Content Based Filtering hanya menggunakan fitur **product_name, category, price, brand, color dan size**, karena Metode ini fokus pada features (atribut) dari produk untuk merekomendasikan produk yang mirip dengan produk yang pernah diinteraksi oleh pengguna. Sedangkan untuk pemodelan Collaborative Filtering menggunakan fitur **user_id, rating, product_name, dan product_id**, karena Metode ini fokus pada pola interaksi antara pengguna dan produk untuk membuat rekomendasi, berdasarkan data eksplisit seperti Rating atau data historis pembelian.

## - Pemodelan
**1. Content Based Filtering**
Dalam pendekatan ini, digunakan metode TF-IDF dan Cosine Similarity untuk memperoleh rekomendasi yang relevan. TF-IDF (Term Frequency-Inverse Document Frequency) untuk fitur product_name menghitung pentingnya kata dalam sebuah dokumen dibandingkan seluruh dokumen di dataset. Nilai TF-IDF tinggi menunjukkan kata yang spesifik dan relevan untuk dokumen tersebut. Langkah pertama dalam pembangunan matriks TF-IDF adalah menghitung Term Frequency (TF), yaitu frekuensi kata dalam deskripsi produk. Selanjutnya, perlu dihitung Inverse Document Frequency (IDF), yaitu kebalikan dari frekuensi kata di seluruh dokumen. Ini mengurangi bobot kata-kata yang umum dan meningkatkan bobot kata-kata yang jarang muncul. Akhirnya, matriks TF-IDF terbentuk dengan mengalikan nilai TF dengan nilai IDF. Matriks ini akan menjadi representasi numerik dari kategori produk dalam dataset. 
Berikutnya menggunakan Cosine Similarity untuk mengukur kemiripan antara dua produk dengan menghitung sudut kosinus antar vektor. Dengan kode : from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
Dengan menghitung cosine similarity antara semua pasangan produk, kita dapat memperoleh matriks similarity antara produk-produk dalam dataset. Nilai cosine similarity ini dapat digunakan untuk menemukan produk-produk yang paling mirip satu sama lain. Semakin tinggi nilai cosine similarity antara dua produk, semakin mirip kedua produk tersebut. 
Output dari pendekatan Content-Based Filtering ini adalah daftar produk rekomendasi berdasarkan keyword yang dimasukan. Misalnya, pengguna aplikasi melakukan pencarian dengan keyword "baju", maka akan ditampilkan beberapa rekomendasi produk yang relevan dengan keyword "baju".
Namun karena data yang digunakan pada proyek ini tidak memiliki nama produk, maka akan diberikan contoh dengan keyword "product_265". Diperoleh top 10 rekomendasi produk berdasarkan keyword tersebut. 

**2. Collaborative Filtering**
Digunakan model RecommerderNet berbasis TensorFlow untuk mempelajari pola preferensi pelanggan dan interaksi mereka dengan produk. Model ini menggunakan embedding untuk merepresentasikan pelanggan dan produk. Dengan menggabungkan embedding tersebut, kami dapat memprediksi preferensi pelanggan terhadap produk tertentu.
Berbeda dengan pendekatan Content-Based Filtering yang hanya menggunakan data informasi tentang produk saja, pendekatan ini akan menggunakan data pelanggan juga. Dataset yang digunakan dibagi menjadi data traning dan data testing dengan rasio 80%:20%.
Output dari pendekatan ini adalah daftar produk rekomendasi untuk setiap pelanggan. Misalnya, pelanggan X akan menerima rekomendasi berupa daftar produk yang banyak disukai oleh pelanggan lain dengan preferensi yang serupa.

## - Referensi
Zhao, X. (2019). A Study on E-commerce Recommender System Based on Big Data. 2019 IEEE 4th International Conference on Cloud Computing and Big Data Analysis (ICCCBDA). doi:https://doi.org/10.1109/icccbda.2019.8725694.
