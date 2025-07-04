import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------------------
# 0. Download Dataset from Google Drive if not present
# ---------------------------
def download_from_drive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, dest_path)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# ---------------------------
# 1. Download Files if Needed
# ---------------------------
os.makedirs("data", exist_ok=True)
files_to_download = {
    "data/BX-Books.csv": "1t-MhJvHceB2brCMinMer-A3HhiUvw8xJ",
    "data/BX-Book-Ratings-Subset.csv": "1TjSUSrNQD2rtpIa3MBqvQnwjVFeOgDMF",
    "data/BX-Users.csv": "1pm_oV9kIKqKrDelP1KA-eQ4oRp6rz7mJ"
}
for path, fid in files_to_download.items():
    if not os.path.exists(path):
        download_from_drive(fid, path)

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("data/BX-Books.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv("data/BX-Book-Ratings-Subset.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv("data/BX-Users.csv", sep=';', encoding='latin-1', on_bad_lines='skip')

    # Clean column names
    books.columns = books.columns.str.strip().str.replace('"', '')
    ratings.columns = ratings.columns.str.strip().str.replace('"', '')
    users.columns = users.columns.str.strip().str.replace('"', '')

    return books, ratings, users


def preprocess_ratings(ratings):
    ratings = ratings[ratings['Book-Rating'] > 0]
    ratings['Book-Rating'] = MinMaxScaler().fit_transform(ratings[['Book-Rating']])
    return ratings

# ---------------------------
# 3. Recommender Logic
# ---------------------------
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
# 4. Display Helper
# ---------------------------
def show_book_tile(column, book):
    with column:
        img_url = book.get('Image-URL-M') or "https://via.placeholder.com/128x193.png?text=No+Image"
        if not isinstance(img_url, str) or not img_url.startswith("http"):
            img_url = "https://via.placeholder.com/128x193.png?text=No+Image"
        st.image(img_url, use_column_width=True)
        st.caption(book['Book-Title'])

# ---------------------------
# 5. Streamlit App UI
# ---------------------------
st.set_page_config(page_title="BookSage 📚", layout="wide")
st.title("📚 BookSage – Your Wise Reading Companion")

books, ratings, users = load_data()
ratings = preprocess_ratings(ratings)

# ---------------------------
# 6. Sidebar Input
# ---------------------------
st.sidebar.header("📖 Customize Recommendation")
input_method = st.sidebar.radio("How would you like to start?", ["Favorite Book", "Genre"])
user_input_book = None

if input_method == "Favorite Book":
    unique_books = books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].drop_duplicates()
    unique_books['Display'] = unique_books.apply(
        lambda row: f"{row['Book-Title']} | {row['Book-Author']} | {row['Year-Of-Publication']} | {row['Publisher']}",
        axis=1
    )
    selected_display = st.sidebar.selectbox("Choose a Book", unique_books['Display'].values)
    selected_parts = selected_display.split('|')
    if len(selected_parts) == 4:
        selected_title = selected_parts[0].strip()
        selected_author = selected_parts[1].strip()
        selected_year = selected_parts[2].strip()
        selected_publisher = selected_parts[3].strip()
        user_input_book = books[
            (books['Book-Title'] == selected_title) &
            (books['Book-Author'] == selected_author) &
            (books['Year-Of-Publication'].astype(str) == selected_year) &
            (books['Publisher'] == selected_publisher)
        ].iloc[0]

elif input_method == "Genre":
    genres = ["Fiction", "Fantasy", "Mystery", "History", "Romance", "Science", "Biography"]
    selected_genre = st.sidebar.selectbox("Pick a Genre", genres)
    genre_books = books[books['Book-Title'].str.contains(selected_genre, case=False, na=False)]
    if not genre_books.empty:
        user_input_book = genre_books.sample(1).iloc[0]
    else:
        st.warning("No books found for this genre.")
        st.stop()

# ---------------------------
# 7. Display Selected Book
# ---------------------------
if user_input_book is not None:
    st.subheader("📖 Selected Book Details")
    col1, col2 = st.columns([1, 3])
    show_book_tile(col1, user_input_book)

    with col2:
        details_df = pd.DataFrame({
            "Field": ["Title", "Author", "Year", "Publisher"],
            "Value": [
                user_input_book.get("Book-Title", "N/A"),
                user_input_book.get("Book-Author", "N/A"),
                user_input_book.get("Year-Of-Publication", "N/A"),
                user_input_book.get("Publisher", "N/A"),
            ]
        })
        st.table(details_df)

    # ---------------------------
    # 8. Recommendations
    # ---------------------------
    st.markdown("## 📚 Recommended for You")
    recommendations = content_based(user_input_book['Book-Title'], books, top_n=5)

    if not recommendations.empty:
        cols = st.columns(len(recommendations))
        for col, book in zip(cols, recommendations.to_dict(orient='records')):
            show_book_tile(col, book)
    else:
        st.info("No similar books found.")

    st.markdown("---")
    st.caption("📚 Powered by BookCrossing | “Books are a uniquely portable magic.” – Stephen King")

