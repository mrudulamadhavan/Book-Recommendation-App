# book_recommender_app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.metrics.pairwise import cosine_similarity

# st.set_page_config(page_title="📚 BookSage – Your Book Recommending Partner", layout="wide")
# st.title("📚 **BookSage – Your Book Recommending Partner**")
st.markdown("<h3 style='text-align: center;'>📚 <b>BookSage – Your Book Recommending Partner  📚</b></h3>", unsafe_allow_html=True)



@st.cache_data
def load_data():
    # Google Drive file IDs
    file_links = {
        "Books.csv": "1zsvB70Z9ga_FMEYR2WvIFHWRFSKcuJUF",
        "Ratings.csv": "1soFmdx-NqJwjWGFfp6xK2-oSTYWPbwnE",
        "Users.csv": "17ZsHcBCNkLxeeMA0YgJZ6rkOgnCasSXV"
    }

    # Download files if not already present
    for filename, file_id in file_links.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)

    # Load datasets
    books = pd.read_csv("Books.csv", encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv("Ratings.csv", encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv("Users.csv", encoding='latin-1', on_bad_lines='skip')

    # Drop missing values in book info
    books.dropna(inplace=True)

    # Clean up 'Year-Of-Publication'
    books = books[books['Year-Of-Publication'].astype(str).str.isnumeric()]
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
    books = books[
        (books['Year-Of-Publication'] >= 1900) &
        (books['Year-Of-Publication'] <= 2023)
    ]

    # Filter users who rated more than 200 books
    active_users = ratings['User-ID'].value_counts()
    active_users = active_users[active_users > 200].index
    ratings = ratings[ratings['User-ID'].isin(active_users)]

    # Merge ratings with books
    rating_with_books = ratings.merge(books, on='ISBN')

    # Filter books with at least 50 ratings
    book_rating_counts = rating_with_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
    book_rating_counts.rename(columns={'Book-Rating': 'number_of_ratings'}, inplace=True)

    final_rating = rating_with_books.merge(book_rating_counts, on='Book-Title')
    final_rating = final_rating[final_rating['number_of_ratings'] >= 50]

    # Remove duplicate ratings
    final_rating.drop_duplicates(['User-ID', 'Book-Title'], inplace=True)

    # Create pivot table: books as rows, users as columns
    book_pivot = final_rating.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    book_pivot.fillna(0, inplace=True)

    # Compute cosine similarity between books
    similarity_scores = cosine_similarity(book_pivot)

    return books, book_pivot, similarity_scores


books, book_pivot, similarity_scores = load_data()

def recommend(book_name):
    if book_name not in book_pivot.index:
        return pd.DataFrame()

    index = np.where(book_pivot.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]  # Skip the book itself

    recommendations = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == book_pivot.index[i[0]]].drop_duplicates('Book-Title')
        if not temp_df.empty:
            recommendations.append({
                'Title': temp_df['Book-Title'].values[0],
                'Author': temp_df['Book-Author'].values[0],
                'Image_URL': temp_df['Image-URL-M'].values[0],
                'Similarity Score': round(i[1], 4)
            })

    return pd.DataFrame(recommendations)

# Streamlit UI
# st.title("📖 Book Recommendation System")
st.write("")
st.markdown("<h4 style='text-align: center;'> <b>Discover your next favorite read effortlessly!</b></h4>", unsafe_allow_html=True)


book_list = list(book_pivot.index)
selected_book = st.selectbox("Select a book you like:", sorted(book_list))
st.write(f"You selected:")

# Show selected book details and image
selected_book_info = books[books['Book-Title'] == selected_book].drop_duplicates('Book-Title')

if not selected_book_info.empty:
    book_img = selected_book_info['Image-URL-M'].values[0]
    book_author = selected_book_info['Book-Author'].values[0]
    book_year = selected_book_info['Year-Of-Publication'].values[0]

    cols = st.columns([1, 3])
    with cols[0]:
        st.image(book_img, width=120)

    with cols[1]:
        st.markdown(f"### {selected_book}")
        st.markdown(f"**Author:** {book_author}")
        st.markdown(f"**Year of Publication:** {book_year}")

if st.button("Recommend"):
    with st.spinner("Finding books you might enjoy..."):
        results = recommend(selected_book)

        if not results.empty:
            st.subheader("📚 Recommended Books for You")

            # Create columns to layout recommendations as tiles
            cols = st.columns(len(results))

            for i, row in results.iterrows():
                with cols[i]:
                    st.image(row['Image_URL'], width=120)
                    st.caption(f"**{row['Title']}**")
                    st.text(f"by {row['Author']}")
        else:
            st.warning("Sorry! Could not find similar books.")

st.markdown("---")
st.caption("📚  “Books are a uniquely portable magic.” – Stephen King")


