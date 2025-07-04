# book_recommender_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, NMF, Dataset, Reader
from surprise.model_selection import train_test_split
import random
import os

# ---------------------------
# 0. Download Dataset from Google Drive if not present
# ---------------------------
def load_data_from_url():
    books_url = "https://drive.google.com/uc?export=download&id=1t-MhJvHceB2brCMinMer-A3HhiUvw8xJ"
    ratings_url = "https://drive.google.com/uc?export=download&id=1MEY18Hr__QtE_Q-19GHe7G6cZ0ABq7uF"
    users_url = "https://drive.google.com/uc?export=download&id=1pm_oV9kIKqKrDelP1KA-eQ4oRp6rz7mJ"
    
    books = pd.read_csv(books_url, sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv(ratings_url, sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(users_url, sep=';', encoding='latin-1', on_bad_lines='skip')
    
    return books, ratings, users

# ---------------------------
# 1. App Config
# ---------------------------
st.set_page_config(page_title="BookSage 📚", layout="wide")
st.title("📚 BookSage – Your Wise Reading Companion")

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("data/BX-Books.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv("data/BX-Book-Ratings-Subset.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv("data/BX-Users.csv", sep=';', encoding='latin-1', on_bad_lines='skip')
    return books, ratings, users

def preprocess_ratings(ratings):
    ratings = ratings[ratings['Book-Rating'] > 0]
    ratings['Book-Rating'] = MinMaxScaler().fit_transform(ratings[['Book-Rating']])
    return ratings

books, ratings, users = load_data()
ratings = preprocess_ratings(ratings)

# ---------------------------
# 3. Helper Functions
# ---------------------------
def show_tile(column, book):
    with column:
        img_url = book.get('Image-URL-M') or "https://via.placeholder.com/128x193.png?text=No+Image"
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

@st.cache_resource
def train_model(model_type="svd", sample_size=10000):
    reader = Reader(rating_scale=(0, 1))
    sample = ratings.sample(sample_size)
    data = Dataset.load_from_df(sample[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD() if model_type == "svd" else NMF()
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = np.sqrt(mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions]))
    mae = mean_absolute_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions])
    return model, rmse, mae

def get_user_recommendations(model, user_id, books, ratings, top_n=5):
    read_books = ratings[ratings['User-ID'] == user_id]['ISBN']
    unread_books = books[~books['ISBN'].isin(read_books)]
    predictions = [
        (isbn, model.predict(user_id, isbn).est)
        for isbn in unread_books['ISBN'].unique()[:500]  # limit for performance
    ]
    top_isbns = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return books[books['ISBN'].isin([isbn for isbn, _ in top_isbns])]

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
# 7. Recommendations
# ---------------------------
st.subheader("🎯 Your High Rated Books")
high_rated_books = ratings[(ratings['User-ID'] == user_id) & (ratings['Book-Rating'] > 0.8)]
top_books = books[books['ISBN'].isin(high_rated_books['ISBN'])].drop_duplicates('Book-Title').head(5)
cols = st.columns(len(top_books)) if not top_books.empty else []
for col, book in zip(cols, top_books.to_dict(orient='records')):
    show_tile(col, book)

st.subheader("🧠 Personalized Recommendations")
model_svd, _, _ = train_model("svd")
user_recs = get_user_recommendations(model_svd, user_id, books, ratings)
cols = st.columns(len(user_recs))
for col, book in zip(cols, user_recs.to_dict(orient='records')):
    show_tile(col, book)

st.subheader("🧾 Content-Based Suggestions")
cb_recs = content_based(current_book['Book-Title'], books, top_n=5)
cols = st.columns(len(cb_recs))
for col, book in zip(cols, cb_recs.to_dict(orient='records')):
    show_tile(col, book)

st.markdown("---")
st.caption("📚 Powered by BookCrossing | “Books are a uniquely portable magic.” – Stephen King")
