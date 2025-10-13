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

