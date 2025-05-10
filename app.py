import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies_2024.csv")
df.dropna(inplace=True)

# Add dummy genres and poster links
df['Genre'] = ["Fantasy", "Adventure", "Fantasy", "Fantasy", "Adventure"]
df['Poster'] = [
    "https://via.placeholder.com/150x220.png?text=The+Wizard's+Journey",
    "https://via.placeholder.com/150x220.png?text=The+Magic+Academy",
    "https://via.placeholder.com/150x220.png?text=The+Dark+Sorcerer",
    "https://via.placeholder.com/150x220.png?text=Spellbound+Year+One",
    "https://via.placeholder.com/150x220.png?text=Mystic+Trials"
]

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Storyline'])

# Streamlit UI
st.set_page_config(page_title="🎬 IMDb Recommender", layout="wide")
st.title("🎬 IMDb Movie Recommender")
st.write("Enter a movie plot to get 5 similar movie recommendations.")

user_input = st.text_area("Enter Movie Storyline:", height=150)

if st.button("Recommend"):
    if user_input.strip():
        input_vec = tfidf.transform([user_input])
        sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[-5:][::-1]
        results = df.iloc[top_indices]

        for _, row in results.iterrows():
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(row["Poster"], width=150)
                with cols[1]:
                    st.subheader(row["Movie Name"])
                    st.caption(f"🎭 Genre: {row['Genre']}")
                    st.write(row["Storyline"])
                st.markdown("---")
    else:
        st.warning("Please enter a valid movie storyline.")
