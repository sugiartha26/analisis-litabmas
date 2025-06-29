import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import networkx as nx
import io

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
    jenis_filter = st.sidebar.multiselect("Filter Jenis Kegiatan", df['JENIS_KEGIATAN'].unique(), default=df['JENIS_KEGIATAN'].unique())
    perguruan_filter = st.sidebar.multiselect("Filter Perguruan Tinggi", df['NMPT'].unique(), default=df['NMPT'].unique())

    df = df[df['TAHUN'].isin(tahun_filter) & df['JENIS_KEGIATAN'].isin(jenis_filter) & df['NMPT'].isin(perguruan_filter)]

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
        topics = []
        topic_rows = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            sentence = "Topik {}: {}".format(topic_idx+1, ', '.join(top_keywords))
            topics.append((f"Topik {topic_idx+1}", top_keywords))
            topic_rows.append({"Topik": f"Topik {topic_idx+1}", "Kalimat": sentence})
            st.markdown(f"**{sentence}**")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_keywords))
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        # Graph Visualization
        st.subheader("🔗 Visualisasi Topik-Kata")
        G = nx.Graph()
        for topic_name, keywords in topics:
            for kw in keywords:
                G.add_node(kw, label=kw)
                G.add_edge(topic_name, kw)

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray', font_size=10, ax=ax)
        st.pyplot(fig)

        # Export to Excel
        st.subheader("📤 Ekspor Hasil Topik")
        topic_df = pd.DataFrame(topic_rows)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            topic_df.to_excel(writer, index=False, sheet_name="Topik Global")
        st.download_button("Download Topik sebagai Excel", data=buffer.getvalue(), file_name="topik_global.xlsx")

    with tab3:
        st.subheader("👨‍🏫 Topik Berdasarkan Nama Dosen")
        top_dosen = df['NMDSN'].value_counts().head(10).index.tolist()
        selected_dosen = st.selectbox("Pilih Nama Dosen", top_dosen)

        dosen_topics_all = []

        for dosen in df['NMDSN'].unique():
            judul_dosen_all = df[df['NMDSN'] == dosen]['JUDUL'].dropna().tolist()
            if len(judul_dosen_all) >= 3:
                try:
                    vectorizer_dosen = TfidfVectorizer(stop_words=stopwords_id, min_df=1, max_df=1.0)
                    tfidf_temp = vectorizer_dosen.fit_transform(judul_dosen_all)
                    nmf_temp = NMF(n_components=1, random_state=42).fit(tfidf_temp)
                    top_kata = [vectorizer_dosen.get_feature_names_out()[i] for i in nmf_temp.components_[0].argsort()[:-11:-1]]
                    sentence = f"{', '.join(top_kata)}"
                    dosen_topics_all.append({"Nama Dosen": dosen, "Topik": sentence})
                except ValueError:
                    continue

        if selected_dosen:
            judul_dosen = df[df['NMDSN'] == selected_dosen]['JUDUL'].dropna().tolist()
            if len(judul_dosen) >= 3:
                try:
                    vectorizer_selected = TfidfVectorizer(stop_words=stopwords_id, min_df=1, max_df=1.0)
                    tfidf_dosen = vectorizer_selected.fit_transform(judul_dosen)
                    nmf_dosen = NMF(n_components=1, random_state=42).fit(tfidf_dosen)
                    top_keywords = [vectorizer_selected.get_feature_names_out()[i] for i in nmf_dosen.components_[0].argsort()[:-11:-1]]
                    sentence = f"Topik Unggulan {selected_dosen}: {', '.join(top_keywords)}"
                    st.markdown(f"**{sentence}**")
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_keywords))
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except ValueError:
                    st.warning("Dosen ini tidak memiliki cukup data untuk topik.")

        if dosen_topics_all:
            st.subheader("📤 Ekspor Semua Topik Dosen")
            df_all_dosen_topics = pd.DataFrame(dosen_topics_all)
            buffer_all = io.BytesIO()
            with pd.ExcelWriter(buffer_all, engine='openpyxl') as writer:
                df_all_dosen_topics.to_excel(writer, index=False, sheet_name="Topik Semua Dosen")
            st.download_button("Download Semua Topik Dosen", data=buffer_all.getvalue(), file_name="topik_semua_dosen.xlsx")
        else:
            st.warning("Tidak cukup data untuk membuat topik dosen.")
