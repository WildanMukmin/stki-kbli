from flask import Flask, render_template, request, abort
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# 1. Load Dataset KBLI yang sudah dipreprocess
# Pastikan nama filenya sesuai dengan yang kamu miliki
df = pd.read_csv('kbli_preprocessed.csv')
df['kode'] = df['kode'].astype(str)

# 2. Load Model TF-IDF dan Matrix dari file .pkl
print("Memuat model .pkl dan matriks TF-IDF...")
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix_kbli = pickle.load(f)
print("Semua model dan dataset siap digunakan!")

# 3. Inisialisasi Stemmer Sastrawi (Hanya digunakan untuk query user saja)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_query(text):
    """Membersihkan dan melakukan stemming pada query input dari user"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return stemmer.stem(text)


# --- ROUTING FLASK ---

# Page 1: Google Home Screen
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Page 2: Google Search Result Page (SERP)
@app.route('/search', methods=['POST'])
def search():
    user_query = request.form.get('query', '')
    if not user_query.strip():
        return render_template('index.html', error="Input pencarian tidak boleh kosong!")

    # Preprocess query yang masuk
    clean_query = preprocess_query(user_query)
    query_words = set(clean_query.split())
    
    # Transform query menggunakan vectorizer hasil load dari .pkl
    tfidf_matrix_query = vectorizer.transform([clean_query])
    
    # Hitung Cosine Similarity antara query dengan matriks KBLI (.pkl)
    similarity_scores = cosine_similarity(tfidf_matrix_query, tfidf_matrix_kbli).flatten()
    
    df_result = df.copy()
    df_result['score'] = similarity_scores
    
    # Ambil 5 teratas yang skor kemiripannya di atas 0
    top_results = df_result[df_result['score'] > 0].sort_values(by='score', ascending=False).head(5)
    
    results = []
    for idx, row in top_results.iterrows():
        # Ambil kolom teks yang sudah bersih dari kbli_preprocessed.csv
        # Sesuaikan nama kolomnya (misal: 'clean_features', 'text_clean', dll.)
        clean_kbli_text = row['clean_features'] if 'clean_features' in row else ''
        kbli_words = set(str(clean_kbli_text).split())
        matched_tokens = query_words.intersection(kbli_words)
        
        results.append({
            'kode': row['kode'],
            'judul': row['judul'],
            'deskripsi': row['deskripsi'],
            'score': row['score'],
            'matched_words': ', '.join(matched_tokens) if matched_tokens else 'Tidak ada'
        })
    
    return render_template('results.html', query=user_query, results=results)


# Page 3: Detail Halaman KBLI saat Tautan Diklik
@app.route('/kbli/<kode>', methods=['GET'])
def kbli_detail(kode):
    detail_data = df[df['kode'] == str(kode)]
    if detail_data.empty:
        abort(404)
        
    selected_kbli = detail_data.iloc[0].to_dict()
    
    # Menyesuaikan nama kolom teks preprocessed jika ingin ditampilkan di halaman detail
    selected_kbli['clean_features'] = selected_kbli.get('clean_features', '')
    
    return render_template('detail.html', kbli=selected_kbli)


if __name__ == '__main__':
    app.run(debug=True)