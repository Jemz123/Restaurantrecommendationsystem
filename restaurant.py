import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample dataset of restaurants
data = {
    'restaurant_name': ['The Pizza Place', 'Sushi World', 'Burger King', 'Vegan Delights', 'Pasta Heaven'],
    'cuisine': ['Italian', 'Japanese', 'American', 'Vegan', 'Italian'],
    'location': ['Downtown', 'Uptown', 'Midtown', 'Downtown', 'Uptown'],
    'price_range': ['$$', '$$$', '$$', '$$', '$$$'],
    'rating': [4.5, 4.8, 3.9, 4.3, 4.7],  # Average ratings (1-5)
    'reviews': [200, 150, 300, 50, 400]  # Number of reviews
}

# Create a DataFrame
df = pd.DataFrame(data)

# Combine relevant columns to create a single feature for each restaurant
df['features'] = df['cuisine'] + ' ' + df['location'] + ' ' + df['price_range']

# Content-based filtering using TF-IDF Vectorizer and Cosine Similarity
def content_based_recommendations(restaurant_name):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = df[df['restaurant_name'] == restaurant_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    restaurant_indices = [i[0] for i in sim_scores]
    recommended_restaurants = df['restaurant_name'].iloc[restaurant_indices]
    return recommended_restaurants

# Collaborative filtering using user ratings
def collaborative_filtering_recommendations(user_ratings):
    reader = Reader(rating_scale=(1, 5))
    ratings_data = pd.DataFrame({
        'user': ['User1', 'User2', 'User3', 'User4', 'User5'],
        'restaurant_name': ['The Pizza Place', 'Sushi World', 'Burger King', 'Vegan Delights', 'Pasta Heaven'],
        'rating': user_ratings
    })
    
    data = Dataset.load_from_df(ratings_data[['user', 'restaurant_name', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    model = SVD()
    model.fit(trainset)
    
    predictions = model.test(testset)
    accuracy.rmse(predictions)
    
    restaurant_ratings = {}
    for restaurant in ratings_data['restaurant_name'].unique():
        predicted_rating = model.predict('User1', restaurant).est  # Predict for a specific user (e.g., User1)
        restaurant_ratings[restaurant] = predicted_rating
    
    sorted_restaurants = sorted(restaurant_ratings.items(), key=lambda x: x[1], reverse=True)
    recommended_restaurants = [restaurant[0] for restaurant in sorted_restaurants[:3]]
    return recommended_restaurants

# Example usage:
restaurant_name = 'The Pizza Place'

# Content-based recommendation
print("Content-based recommendations:")
content_recommendations = content_based_recommendations(restaurant_name)
print(content_recommendations)

# Collaborative filtering recommendation (example user ratings)
user_ratings = [4, 5, 2, 3, 4]  # Ratings given by User1 to each restaurant
print("\nCollaborative filtering recommendations:")
collab_recommendations = collaborative_filtering_recommendations(user_ratings)
print(collab_recommendations)
