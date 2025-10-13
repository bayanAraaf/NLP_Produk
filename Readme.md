# 🗣️ Yelp Review Sentiment Analysis

Proyek ini bertujuan untuk menganalisis sentimen dari review pengguna Yelp menggunakan pendekatan supervised learning berbasis teks. Dataset sangat besar (±4GB), sehingga dilakukan sampling, preprocessing, modeling, dan visualisasi untuk membangun sistem klasifikasi dan chatbot sederhana.

---

## 📁 Dataset

- **Sumber**: [Kaggle - Yelp Open Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
- **File**: `review.json`
- **Deskripsi**: Dataset berisi jutaan review pengguna, rating bintang, dan metadata bisnis dari platform Yelp.
- **Lisensi**: Tersedia untuk keperluan riset dan non-komersial.

---

## ⚙️ Tools & Teknologi

- Python (Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud)
- TF-IDF Vectorizer untuk ekstraksi fitur teks
- Logistic Regression untuk klasifikasi
- Joblib untuk menyimpan model
- Jupyter Notebook / VS Code

---

## 🎯 Tujuan Analisis

- Sampling data besar menggunakan reservoir sampling
- Labeling sentimen berdasarkan rating bintang
- Preprocessing teks (tokenisasi, stopwords, lemmatization)
- Ekstraksi fitur dengan TF-IDF
- Training dan evaluasi model klasifikasi
- Visualisasi distribusi dan kata dominan
- Pembuatan chatbot sederhana berbasis model sentimen

---

## 🔍 Langkah Kerja

1. **Sampling & Loading**
   - Mengambil 50.000 review acak dari file JSON besar
   - Memilih kolom `text` dan `stars`

2. **Labeling Sentimen**
   - ⭐ 4–5 → `positive`
   - ⭐ 3 → `neutral`
   - ⭐ 1–2 → `negative`

3. **Preprocessing Teks**
   - Tokenisasi, POS tagging, lemmatization
   - Stopwords filtering (termasuk kata umum seperti "food", "place", "restaurant")

4. **EDA & Visualisasi**
   - Distribusi rating dan sentimen
   - Histogram panjang review
   - Pie chart distribusi sentimen dan rating
   - WordCloud untuk review positif dan negatif

5. **Modeling**
   - TF-IDF vectorization (max_features=20.000, ngram 1–2)
   - Logistic Regression dengan `class_weight="balanced"`
   - Evaluasi: Accuracy, Classification Report, Confusion Matrix

6. **Interpretasi**
   - Top 10 kata paling berpengaruh untuk masing-masing kelas sentimen

7. **Chatbot**
   - Fungsi `chatbot_response()` untuk mengklasifikasikan input teks dan memberi respons sesuai sentimen

8. **Model Persistence**
   - Menyimpan model dan vectorizer dengan `joblib.dump()`

---

## 📈 Visualisasi

- Bar chart distribusi rating dan sentimen
- Histogram panjang review
- Pie chart distribusi sentimen dan rating
- WordCloud untuk kata dominan di review positif dan negatif
- Confusion Matrix model klasifikasi

---

## 🧠 Insight

- Mayoritas review bersentimen positif
- Kata-kata seperti “great”, “amazing”, “friendly” dominan di review positif
- Kata-kata seperti “bad”, “slow”, “rude” dominan di review negatif
- Logistic Regression cukup efektif untuk klasifikasi teks dengan preprocessing yang baik

---

## 🚀 Cara Menjalankan

1. Pastikan semua dependensi terinstal:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud joblib

2.  Unduh dan ekstrak dataset Yelp dari Kaggle

3. Jalankan Notebook:
   jupyter notebook yelp_sentiment_analysis.ipynb

4. Unduh resource NLTK (cukup sekali):
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')

📎 Attribution & Credits
- Dataset: Yelp Open Dataset oleh Yelp Inc. via Kaggle
- Lisensi: Untuk keperluan riset dan non-komersial
- Pengembangan Proyek: Oleh Bayan sebagai eksplorasi klasifikasi teks, NLP, dan chatbot berbasis sentimen


## Readme 2

# 🤖 Yelp Review Sentiment Chatbot (Streamlit App)

Aplikasi ini adalah chatbot interaktif berbasis Streamlit yang memprediksi sentimen dari ulasan pengguna Yelp dan memberikan respons otomatis. Model klasifikasi dilatih dari dataset besar Yelp dan menggunakan teknik NLP modern seperti TF-IDF dan Logistic Regression.

---

## 📁 Dataset

- **Sumber**: [Kaggle - Yelp Open Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
- **File**: `review.json`
- **Deskripsi**: Dataset berisi jutaan review pengguna, rating bintang, dan metadata bisnis dari platform Yelp.
- **Lisensi**: Untuk keperluan riset dan non-komersial.

---

## ⚙️ Tools & Teknologi

- Python (Pandas, NumPy, Scikit-learn, NLTK, Streamlit, Matplotlib, Seaborn, WordCloud, Joblib)
- TF-IDF Vectorizer untuk ekstraksi fitur teks
- Logistic Regression untuk klasifikasi
- Streamlit untuk antarmuka pengguna
- Joblib untuk menyimpan dan memuat model

---

## 🎯 Fitur Aplikasi

- Input teks ulasan dari pengguna
- Prediksi sentimen: `positive`, `neutral`, atau `negative`
- Probabilitas klasifikasi dalam bentuk bar chart
- Respon chatbot otomatis berdasarkan hasil prediksi
- Visualisasi distribusi sentimen dan rating
- WordCloud untuk kata dominan di review positif dan negatif

---

## 🔍 Alur Kerja

1. **Sampling & Preprocessing**
   - Reservoir sampling 50.000 review dari file JSON besar
   - Labeling sentimen berdasarkan rating bintang
   - Pembersihan teks: tokenisasi, stopwords, lemmatization

2. **Modeling**
   - TF-IDF vectorization (max_features=20.000, ngram 1–2)
   - Logistic Regression (`class_weight="balanced"`, `max_iter=2000`)
   - Evaluasi: Accuracy, Classification Report, Confusion Matrix
   - Interpretasi: Top kata per kelas sentimen

3. **Chatbot**
   - Fungsi `chatbot_response()` untuk klasifikasi dan respons otomatis
   - Respons berbeda untuk sentimen positif, netral, dan negatif

4. **Streamlit App**
   - Input teks dari pengguna
   - Prediksi sentimen dan probabilitas
   - Visualisasi bar chart
   - Respons chatbot langsung di layar

---

## 🧪 Cara Menjalankan

1. **Instalasi dependensi**
   ```bash
   pip install streamlit scikit-learn pandas numpy nltk matplotlib seaborn wordcloud joblib

2. Unduh resource NLTK (sekali saja)
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')

3. Jalankan aplikasi Streamlit
   cd "C:\Users\HYPE AMD\latihan"
   python -m streamlit run jupiter/project/project3.py


📎 Attribution & Credits- Dataset: Yelp Open Dataset oleh Yelp Inc. via Kaggle
- Model & Vectorizer: Dilatih dari proyek analisis sentimen Yelp oleh Bayan
- Pengembangan Aplikasi: Oleh Bayan sebagai eksplorasi NLP, klasifikasi teks, dan chatbot interaktif berbasis Streamlit

