from flask import Flask, render_template, request, abort
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

app = Flask(__name__)

# 1. Load Data KBLI
df = pd.read_csv('dataset.csv')
# Pastikan kolom kode dibaca sebagai string agar tidak bermasalah di URL
df['kode'] = df['kode'].astype(str)

# 2. Inisialisasi Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return stemmer.stem(text)

print("Sedang memproses teks dataset KBLI...")
df['clean_features'] = (df['judul'] + " " + df['deskripsi']).apply(preprocess_text)

# 3. Fit Model TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix_kbli = vectorizer.fit_transform(df['clean_features'])
print("Dataset siap!")

# --- ROUTING FLASK ---

# Page 1: Search Box (Google Home)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Page 2: Serp (Google Search Result Page)
@app.route('/search', methods=['POST'])
def search():
    user_query = request.form.get('query', '')
    if not user_query.strip():
        return render_template('index.html', error="Input pencarian tidak boleh kosong!")

    clean_query = preprocess_text(user_query)
    query_words = set(clean_query.split())
    
    tfidf_matrix_query = vectorizer.transform([clean_query])
    similarity_scores = cosine_similarity(tfidf_matrix_query, tfidf_matrix_kbli).flatten()
    
    df_result = df.copy()
    df_result['score'] = similarity_scores
    
    top_results = df_result[df_result['score'] > 0].sort_values(by='score', ascending=False).head(5)
    
    results = []
    for idx, row in top_results.iterrows():
        kbli_words = set(row['clean_features'].split())
        matched_tokens = query_words.intersection(kbli_words)
        
        results.append({
            'kode': row['kode'],
            'judul': row['judul'],
            'deskripsi': row['deskripsi'],
            'score': row['score'],
            'matched_words': ', '.join(matched_tokens) if matched_tokens else 'Tidak ada'
        })
    
    return render_template('results.html', query=user_query, results=results)

# Page 3: Detail Halaman KBLI saat Link diklik
@app.route('/kbli/<kode>', methods=['GET'])
def kbli_detail(kode):
    # Cari data di dataframe berdasarkan kode yang diklik
    detail_data = df[df['kode'] == str(kode)]
    
    if detail_data.empty:
        abort(404) # Jika kode tidak ditemukan tampilkan error 404
        
    # Ambil baris pertama data yang cocok
    selected_kbli = detail_data.iloc[0].to_dict()
    
    return render_template('detail.html', kbli=selected_kbli)

if __name__ == '__main__':
    app.run(debug=True)