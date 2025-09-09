# Klasifikasi MRI 
## Penjelasan Proyek
Proyek ini merupakan proyek klasifikasi MRI yang terdiri dari empat kelas, yaitu **Mild Demented**, **Moderate Demented**, **Non Demented**, dan **Very Mild Demented**. Dataset diperoleh dari Kaggle dan dapat diakses melalui tautan [berikut](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset). 

## Distribusi Gambar
Berikut ini adalah jumlah gambar per kelas:

| Kelas | Jumlah |
| ------ | ------ |
| VeryMildDemented | 8960 |
| ModerateDemented | 6464 |
| NonDemented | 9600 |
| MildDemented | 8960 |
| **Total** | **33984** |

## Arsitektur Model
1. **VGG16 Pre-Trained Model**
    - Menggunakan VGG16, arsitektur transfer learning yang telah terlatih pada dataset ImageNet.
    - Ukuran input model yang digunakan adalah (150 x 150 x 3), sesuai dengan dimensi gambar dalam dataset.
2. **Menggunakan 20 Layer Terakhir dari Model VGG16**
Hanya 20 layer terakhir dari VGG16 yang difinetune, sementara layer sebelumnya dibekukan untuk mempertahankan fitur yang telah dipelajari.
3. **Conv2D**
Menambahkan lapisan Conv2D dengan 32 filter dan 64 filter berukuran (3,3) dan fungsi aktivasi ReLU serta padding 'same'. Setelah itu, model dilanjutkan dengan lapisan MaxPooling2D berukuran (2,2).
5. **Flatten Layer**
Menggunakan GlobalAveragePooling2D untuk mereduksi dimensi sebelum masuk ke fully connected layer.
6. **Fully Connected Layer**
    - Menambahkan Dropout layer (0.5) untuk mengurangi risiko overfitting.
    - Dense layer dengan 256 unit dan aktivasi ReLU 
    - Output layer menggunakan Dense layer dengan 4 unit (sesuai jumlah kelas) dan aktivasi softmax untuk menghasilkan probabilitas tiap kelas.
7. **Kompilasi Model**
    - Menggunakan **Adam optimizer** dengan learning rate sebesar 1e-4 untuk optimasi parameter.
    - Loss function yang digunakan adalah **categorical crossentropy**, sesuai untuk klasifikasi multi-kelas.
    - Metrik evaluasi utama adalah akurasi untuk mengukur kinerja model dalam membedakan kelas MRI.
8. **Callback**
Callback digunakan untuk mengontrol proses pelatihan agar lebih optimal dan mencegah overfitting. Callback yang digunakan antara lain:
    - **Custom Callback (myCallback)**
        Menghentikan pelatihan jika accuracy dan val_accuracy telah mencapai lebih dari 95%.
    - **Model Checkpoint**
        Menyimpan model terbaik berdasarkan nilai val_loss terkecil.
    - **Early Stopping**
            - Menghentikan pelatihan jika val_accuracy tidak mengalami peningkatan dalam 5 epoch berturut-turut.
            - **restore_best_weights=True** akan mengembalikan bobot terbaik sebelum training berhenti.



## Train Model

Model dilatih selama 20 epoch, namun karena adanya callback, pelatihan berhenti pada epoch ke-5 karena model telah mencapai akurasi 95% pada data pelatihan dan evaluasi. Model terbaik kemudian disimpan dalam file best_model.h5.

| **epoch** | **loss** | **accuracy** | **val_loss** | **val_accuracy** |
| --------  | -------- |  --------    |   ---------  |  --------        |
| 1/20      | 1.0480   | 0.4847       | 0.6644       |  0.6787          |
| 2/20      | 0.6233   | 0.7061       | 0.5672       |  0.7506          |
| 3/20      | 0.4110   | 0.8225       | 0.2435       |  0.8976          |
| 4/20      | 0.2131   | 0.9164       | 0.1083       |  0.9623          |
| 5/20      | 0.1295   | 0.9501       | 0.0814       |  0.9697          |

### Grafik Akurasi dan Loss


![accuracy plot](https://github.com/user-attachments/assets/199f269e-0ec8-4759-aa10-018ed6f54f8f)

![loss plot](https://github.com/user-attachments/assets/a000e1d4-d49c-4fd7-a64b-2619efd78166)

## How To Run

Inference menggunakan TF Serving

- Buka Docker Desktop
- Lalu unduh (pull) image TensorFlow Serving dari Docker Hub dengan perintah berikut:
    ```sh
    docker pull tensorflow/serving
    ```
- Install Tensorflow Serving API
    ```sh
    pip install tensorflow-serving-api
    ```
- Buka terminal dengan menjalankan perintah berikut:
    ```sh
    docker run -it -v YOUR_PATH\models:/models -p 8501:8501 --entrypoint /bin/bash tensorflow/serving
    ```
- Menjalankan model pada Docker container dengan perintah berikut:
    ```sh
    tensorflow_model_server --rest_api_port=8501 --model_name=klasifikasi_mri --model_base_path=/saved_model/klasifikasi_mri/
    ```
- Ujilah hasil deployment dengan menyalin URL pada alamat web browser. 
    ```sh
    (http://localhost:8501/v1/moodels/klasifikasi_mri) 
    ```

Atau dapat langsung mengakses web dengan link [berikut](https://alzheimer-disease.streamlit.app/)
