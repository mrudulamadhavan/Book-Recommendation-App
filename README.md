
# 📚 Book Recommendation System

## Overview
Discover your next favorite read effortlessly! This Book Recommendation System leverages the power of **collaborative filtering** to connect you with personalized book suggestions based on user preferences and reading history.

## Features
- Content-based filtering with book metadata
- Collaborative filtering using user-book ratings
- Book similarity calculation using cosine similarity
- Clean and interactive UI via Streamlit
- Google Drive integration for dataset handling

## Dataset
Book-Crossing Dataset with the following files:
- `Books.csv` – Book details (title, author, image, etc.)
- `Ratings.csv` – User ratings for books
- `Users.csv` – User demographics

## Preprocessing Highlights
- Filtered active users (rated > 200 books)
- Filtered popular books (rated by > 50 users)
- Cleaned book data (e.g., publication year normalization)

## Recommendation Algorithm
Collaborative filtering using a pivot table (books x users) and cosine similarity to find similar titles based on user ratings.

## How to Run
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run Streamlit app:
    ```bash
    streamlit run book_recommender_app.py
    ```

## Evaluation and Limitations
### Evaluation Metrics (for future enhancement):
- RMSE, MAE for rating prediction models
- Precision@K, Recall@K for ranking evaluation

### Limitations:
- Relies on explicit feedback only
- Cold start problem for new users/items
- Popularity bias for books with fewer ratings

## Quotes & UX
> “Books are a uniquely portable magic.” – Stephen King


### **Streamlit Link: ** https://recommendbookapp.streamlit.app/



