import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------------------
# 0. Load Dataset from Google Drive URLs
# ---------------------------
@st.cache_data
def load_data_from_url():
    books_url = "https://drive.google.com/uc?export=download&id=1t-MhJvHceB2brCMinMer-A3HhiUvw8xJ"
    ratings_url = "https://drive.google.com/uc?export=download&id=1MEY18Hr__QtE_Q-19GHe7G6cZ0ABq7uF"
    users_url = "https://drive.google.com/uc?export=download&id=1pm_oV9kIKqKrDelP1KA-eQ4oRp6rz7mJ"

    books = pd.read_csv(books_url, sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv(ratings_url, sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(users_url, sep=';', encoding='latin-1', on_bad_lines='skip')

    return books, ratings, users

def preprocess_ratings(ratings):
    ratings = ratings[ratings['Book-Rating'] > 0]
    ratings['Book-Rating'] = MinMaxScaler().fit_transform(ratings[['Book-Rating']])
    return ratings

# ---------------------------
# 1. App Config
# ---------------------------
st.set_page_config(page_title="BookSage 📚", layout="wide")
st.title("📚 BookSage – Your Wise Reading Companion")

# ---------------------------
# 2. Load and Prepare Data
# ---------------------------
books, ratings, users = load_data_from_url()
ratings = preprocess_ratings(ratings)

# ---------------------------
# 3. Helper Functions
# ---------------------------
def show_tile(column, book):
    with column:
        img_url = book.get('Image-URL-M') or "https://via.placeholder.com/128x193.png?text=No+Image"
        if not isinstance(img_url, str) or not img_url.startswith("http"):
            img_url = "https://via.placeholder.com/128x193.png?text=No+Image"
        st.image(img_url, use_column_width=True)
        st.caption(book['Book-Title'])

def content_based(book_title, books_df, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    books_df['Book-Title'] = books_df['Book-Title'].fillna('')
    tfidf_matrix = tfidf.fit_transform(books_df['Book-Title'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_df.index, index=books_df['Book-Title']).drop_duplicates()
    idx = indices.get(book_title)
    if idx is None:
        return pd.DataFrame()
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices]

# ---------------------------
# 4. Session Init
# ---------------------------
if 'ISBN' not in st.session_state:
    st.session_state['ISBN'] = '0385486804'
if 'User-ID' not in st.session_state:
    st.session_state['User-ID'] = 98783

# ---------------------------
# 5. Sidebar
# ---------------------------
st.sidebar.header("📖 Customize")
user_id = st.sidebar.number_input("User ID", min_value=1, value=st.session_state['User-ID'])
st.session_state['User-ID'] = user_id
start_method = st.sidebar.radio("Start with:", ["Favorite Book", "Genre"])

if start_method == "Favorite Book":
    fav_book = st.sidebar.selectbox("Choose a Book", books['Book-Title'].dropna().unique())
    selected_isbn = books[books['Book-Title'] == fav_book]['ISBN'].values[0]
    st.session_state['ISBN'] = selected_isbn
else:
    genres = ["Fiction", "Fantasy", "Mystery", "History", "Romance", "Science", "Biography"]
    selected_genre = st.sidebar.selectbox("Pick a Genre", genres)
    genre_books = books[books['Book-Title'].str.contains(selected_genre, case=False, na=False)]
    if not genre_books.empty:
        st.session_state['ISBN'] = genre_books.sample(1).iloc[0]['ISBN']
    else:
        st.warning("No books found for this genre.")
        st.stop()

# ---------------------------
# 6. Book Detail
# ---------------------------
current_book = books[books['ISBN'] == st.session_state['ISBN']].iloc[0]
rating_data = ratings[ratings['ISBN'] == st.session_state['ISBN']]
avg_rating = rating_data['Book-Rating'].mean()

col1, col2 = st.columns([1, 3])
with col1:
    st.image(current_book['Image-URL-L'], use_column_width=True)
with col2:
    st.subheader(current_book['Book-Title'])
    st.markdown(f"**Author:** {current_book['Book-Author']}")
    st.markdown(f"**Publisher:** {current_book['Publisher']} ({current_book['Year-Of-Publication']})")
    st.markdown(f"**Average Rating:** {avg_rating:.2f} ⭐" if not pd.isna(avg_rating) else "No ratings yet")

# ---------------------------
# 7. Content-Based Recommendations
# ---------------------------
st.subheader("🧾 Content-Based Recommendations")
cb_recs = content_based(current_book['Book-Title'], books, top_n=5)
if not cb_recs.empty:
    cols = st.columns(len(cb_recs))
    for col, book in zip(cols, cb_recs.to_dict(orient='records')):
        show_tile(col, book)
else:
    st.info("No similar books found.")

st.markdown("---")
st.caption("📚 Powered by BookCrossing | “Books are a uniquely portable magic.” – Stephen King")
