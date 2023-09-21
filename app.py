from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__, template_folder='templates', static_folder='static')

# Define file paths to your datasets
RATINGS_FILE_PATH = 'C:/Users/alyan/Desktop/movie2/ratingsss.csv'
MOVIES_FILE_PATH = 'C:/Users/alyan/Desktop/movie2/movies.csv'

# Load movie and rating datasets
movies_df = pd.read_csv(MOVIES_FILE_PATH)
ratings_df = pd.read_csv(RATINGS_FILE_PATH)

# Create TF-IDF vectorizer and other data needed for movie recommendations
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['title'])

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, _ = train_test_split(data, test_size=0.001)  # Adjust test_size as needed

sim_options = {
    'name': 'cosine',
    'user_based': True
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)


# Function to get genres and ratings for a movie
def get_movie_details(movie_id):
    movie = movies_df[movies_df['movieId'] == movie_id]
    genres = movie['genres'].values[0]
    avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean()
    return genres, avg_rating


# Function to get hybrid recommendations with genres and ratings
def hybrid_recommendations(user_id, movie_title, tfidf_matrix, movies_df, model, ratings_df):
    cf_recommendations = []
    for movie_id in ratings_df['movieId'].unique():
        if not ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id)].empty:
            continue
        prediction = model.predict(user_id, movie_id)
        cf_recommendations.append((movie_id, prediction.est))
    cf_recommendations = sorted(cf_recommendations, key=lambda x: x[1], reverse=True)[:10]

    # Content-based recommendations
    idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    content_recommendations = movies_df['title'].iloc[movie_indices].tolist()

    # Hybrid recommendations with genres and ratings
    hybrid_recommendations = []
    for movie_id, _ in cf_recommendations:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        genres, avg_rating = get_movie_details(movie_id)
        hybrid_recommendations.append({
            'title': title,
            'genres': genres,
            'avg_rating': avg_rating
        })

    for title in content_recommendations:
        if title not in hybrid_recommendations:
            genres, avg_rating = get_movie_details(
                movies_df[movies_df['title'] == title]['movieId'].values[0]
            )
            hybrid_recommendations.append({
                'title': title,
                'genres': genres,
                'avg_rating': avg_rating
            })

    return hybrid_recommendations


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['user_id'])
    movie_title = request.form['movie_title']

    recommendations = hybrid_recommendations(user_id, movie_title, tfidf_matrix, movies_df, model, ratings_df)

    return render_template('recommendations.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
