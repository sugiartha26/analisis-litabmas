import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import networkx as nx

st.set_page_config(layout="wide")
st.title("📊 Analisis Kegiatan Penelitian dan Pengabdian Masyarakat")

# Load data
uploaded_file = st.file_uploader("Upload file Excel", type="xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    def clean_currency(val):
        if isinstance(val, str):
            val = val.replace('.', '').replace(',', '.')
            try:
                return float(val)
            except ValueError:
                return 0.0
        return float(val) if pd.notnull(val) else 0.0

    df['DANAPT'] = df['DANAPT'].apply(clean_currency)
    df['DANA_INST_LAIN'] = df['DANA_INST_LAIN'].apply(clean_currency)
    df['TOTAL_DANA'] = df['DANAPT'] + df['DANA_INST_LAIN']

    df['TAHUN'] = df['THNPELAK']
    df['JENIS_KEGIATAN'] = df['JENIS_LITABMAS'].map({'M': 'Pengabdian', 'L': 'Penelitian'})

    # Sidebar filter
    tahun_filter = st.sidebar.multiselect("Filter Tahun", sorted(df['TAHUN'].unique()), default=sorted(df['TAHUN'].unique()))
    df = df[df['TAHUN'].isin(tahun_filter)]

    tab1, tab2, tab3 = st.tabs(["📈 Statistik", "🧠 Topik Global", "👨‍🏫 Topik per Dosen"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribusi Kegiatan per Tahun")
            fig, ax = plt.subplots()
            df['TAHUN'].value_counts().sort_index().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Jenis Kegiatan")
            fig, ax = plt.subplots()
            df['JENIS_KEGIATAN'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Top 10 Perguruan Tinggi")
            st.bar_chart(df['NMPT'].value_counts().head(10))

        with col4:
            st.subheader("Top 10 Skema")
            st.bar_chart(df['SKIM'].value_counts().head(10))

    with tab2:
        st.subheader("🧠 Topik Global dari Judul")
        judul_data = df['JUDUL'].dropna().reset_index(drop=True)

        stopwords_id = [
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'ini',
            'itu', 'dalam', 'atau', 'oleh', 'juga', 'sebagai', 'karena', 'tidak', 'dapat', 'akan',
            'agar', 'lebih', 'bagi', 'terhadap'
        ]

        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stopwords_id)
        tfidf = vectorizer.fit_transform(judul_data)

        nmf_model = NMF(n_components=5, random_state=42)
        nmf_model.fit(tfidf)

        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            st.markdown(f"**Topik {topic_idx+1}:** {', '.join(top_keywords)}")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_keywords))
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

    with tab3:
        st.subheader("👨‍🏫 Topik Berdasarkan Nama Dosen")
        top_dosen = df['NMDSN'].value_counts().head(10).index.tolist()
        selected_dosen = st.selectbox("Pilih Nama Dosen", top_dosen)

        judul_dosen = df[df['NMDSN'] == selected_dosen]['JUDUL'].dropna().tolist()
        if len(judul_dosen) >= 3:
            tfidf_dosen = vectorizer.fit_transform(judul_dosen)
            nmf_dosen = NMF(n_components=1, random_state=42).fit(tfidf_dosen)
            top_keywords = [vectorizer.get_feature_names_out()[i] for i in nmf_dosen.components_[0].argsort()[:-11:-1]]
            st.markdown(f"**Topik Unggulan {selected_dosen}:** {', '.join(top_keywords)}")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_keywords))
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Dosen ini belum memiliki cukup judul untuk dianalisis.")
