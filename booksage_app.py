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



def download_from_drive(file_id, dest_path):
    import gdown
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

# Download datasets if not present
os.makedirs("data", exist_ok=True)
download_from_drive("1t-MhJvHceB2brCMinMer-A3HhiUvw8xJ", "data/BX-Books.csv")
download_from_drive("1MEY18Hr__QtE_Q-19GHe7G6cZ0ABq7uF", "data/BX-Book-Ratings-Subset.csv")
download_from_drive("1pm_oV9kIKqKrDelP1KA-eQ4oRp6rz7mJ", "data/BX-Users.csv")

st.set_page_config(page_title="BookSage 📚", layout="wide")
st.title("📚 BookSage – Your Wise Reading Companion")

# ---------------------------
# 1. Data Loading & Preprocessing (Week 1)
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
# 2. Utility Functions
# ---------------------------
def show_tile(column, book):
    with column:
        st.button('📖', key=random.random(), on_click=select_book, args=(book['ISBN'],))
        st.image(book['Image-URL-M'], use_column_width=True)
        st.caption(book['Book-Title'])

def select_book(isbn):
    st.session_state['ISBN'] = isbn

def content_based(book_title, books_df, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    books_df['Book-Title'] = books_df['Book-Title'].fillna('')
    tfidf_matrix = tfidf.fit_transform(books_df['Book-Title'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_df.index, index=books_df['Book-Title']).drop_duplicates()
    idx = indices.get(book_title)
    if idx is None: return pd.DataFrame()
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices]

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

# ---------------------------
# 3. Session Initialization
# ---------------------------
if 'ISBN' not in st.session_state:
    st.session_state['ISBN'] = '0385486804'
if 'User-ID' not in st.session_state:
    st.session_state['User-ID'] = 98783

# ---------------------------
# 4. Sidebar – Interaction
# ---------------------------
st.sidebar.header("📖 Customize")
start_method = st.sidebar.radio("Start with:", ["Favorite Book", "Genre"])
user_id = st.sidebar.number_input("User ID", min_value=1, value=st.session_state['User-ID'])
st.session_state['User-ID'] = user_id

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
# 5. Book Display Section
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

st.caption("“A reader lives a thousand lives before he dies.” – George R.R. Martin")

# ---------------------------
# 6. Recommendations – High User Rating
# ---------------------------
st.subheader("🎯 Books You Rated Highly")
high_rated_books = ratings[(ratings['User-ID'] == user_id) & (ratings['Book-Rating'] > 0.8)]
top_books = books[books['ISBN'].isin(high_rated_books['ISBN'])].drop_duplicates('Book-Title').head(5)
if top_books.empty:
    st.info("You haven't rated any books yet.")
else:
    cols = st.columns(len(top_books))
    for col, book in zip(cols, top_books.to_dict(orient='records')):
        show_tile(col, book)
st.caption("“A good book is an event in my life.” – Stendhal")

# ---------------------------
# 7. Recommendations – Cross Genre Picks
# ---------------------------
st.subheader("🌍 Explore More Genres")
genre_list = ["Fiction", "Fantasy", "Mystery", "History", "Romance", "Science", "Biography"]
cur_title = current_book['Book-Title'].lower()
cur_genre = next((g for g in genre_list if g.lower() in cur_title), None)

for genre in genre_list:
    if genre == cur_genre:
        continue
    genre_df = books[books['Book-Title'].str.contains(genre, case=False, na=False)]
    merged = genre_df.merge(ratings, on='ISBN')
    if not merged.empty:
        top = merged.groupby('ISBN').mean(numeric_only=True).sort_values(by='Book-Rating', ascending=False).head(1)
        if not top.empty:
            isbn_top = top.index[0]
            book_row = books[books['ISBN'] == isbn_top]
            st.markdown(f"**Top Pick in {genre}**")
            cols = st.columns(1)
            show_tile(cols[0], book_row.iloc[0])

# ---------------------------
# 8. Model Training & Evaluation
# ---------------------------
with st.expander("🧠 Model Evaluation"):
    col1, col2 = st.columns(2)
    with col1:
        model_svd, rmse_svd, mae_svd = train_model("svd")
        st.success(f"SVD - RMSE: {rmse_svd:.3f}, MAE: {mae_svd:.3f}")
    with col2:
        model_nmf, rmse_nmf, mae_nmf = train_model("nmf")
        st.success(f"NMF - RMSE: {rmse_nmf:.3f}, MAE: {mae_nmf:.3f}")
    st.caption("Model performance evaluated using RMSE & MAE.")

# ---------------------------
# 9. Content-Based Recs (Optional)
# ---------------------------
st.subheader("🧾 Content-Based Suggestions")
cb_recs = content_based(current_book['Book-Title'], books, top_n=5)
if not cb_recs.empty:
    cols = st.columns(5)
    for col, book in zip(cols, cb_recs.to_dict(orient='records')):
        show_tile(col, book)

st.markdown("---")
st.caption("📚 Powered by BookCrossing | “Books are a uniquely portable magic.” – Stephen King")
