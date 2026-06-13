import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from thefuzz import process
import streamlit as st

# Spotify Dataset Loaded
data = pd.read_csv('csv/spotify_tracks.csv')

# Data cleaning and preprocessing
# Clustering using numerical features of the Dataset
numerical_features = ['danceability', 'energy', 'acousticness', 'liveness',
                      'speechiness', 'tempo', 'instrumentalness', 'valence', 'loudness']

data = data.dropna(subset=numerical_features)  # Rows are dropped with missing numerical features

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerical_features])  # Numerical features have been normalized

# Clustering process beginning by using K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)  # Decided that 10 clusters would be the most optimal

# Data  preparation for KNN
# track id and popularity is the only relevant columns needed for user interaction data
# Had this annoying issue where the user item matrix would be empty when trying to fit it into the KNN model
# After much research and struggling, I finally got a solution by dropping empty values in the dataset and converting
# the popularity column before creating my pivot table and fitting it into the KNN Model
# Prepare interaction data, ensuring no missing values
interaction_data = data[['track_id', 'popularity']].dropna()

if interaction_data.empty:
    print("No valid interaction data found!")
else:
    # Convert popularity to integer
    interaction_data['popularity'] = interaction_data['popularity'].astype(int)

    # Create pivot table (force at least one column)
    user_item_matrix = interaction_data.pivot_table(
        index='track_id',
        columns='popularity',
        aggfunc='size',
        fill_value=0
    ).astype(float)

    print("Fixed User-Item Matrix Shape:", user_item_matrix.shape)

    # Fit the KNN model
    if user_item_matrix.shape[1] > 0:
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
        knn.fit(user_item_matrix)
    else:
        print("User-Item Matrix has no valid data!")


# Extra function to convert Song duration into minutes
def duration_minutes(duration_ms):
    minutes = int(duration_ms / 60000)
    seconds = int((duration_ms % 60000) / 1000)
    return f"{minutes}:{seconds:02d}"


# Needed a way to refine my searches, before Searches needed exact spelling and this library seemed very useful
# So now with this implementation, It brings up the closely related track or artist based if search not accurate
def fuzzy_search(query, choices):
    match, score = process.extractOne(query, choices)
    return match if score > 70 else None    # Used a 70% accuracy, can be adjusted


# A recommendation function that is based on the track name or artist name
def recommend_tracks(input_query, data, knn, num_recommendations=20, sort_by=None):
    # Find the matching track(s) based on the input query
    track_match = fuzzy_search(input_query, data['track_name'].tolist())
    artist_match = fuzzy_search(input_query, data['artist_name'].tolist())

    match = data[(data['track_name'] == track_match) | (data['artist_name'] == artist_match)]

    if match.empty:
        st.warning("No matching searches found! Now Showing trending songs instead")
        return data.sort_values(by='popularity', ascending=False).head(num_recommendations)

    # Had an approach that offered a list of the closest tracks to     the search, but caused an issue
    # Decided that automatically choosing the first matching track offers a bit more simplicity
    track_info = match.iloc[0]

    track_id = track_info['track_id']
    cluster_id = track_info['cluster']
    track_language = track_info['language']     # Added a language filter because of multiple languages in the dataset

    # Start of filtering process from the same cluster that also checks for the same language
    cluster_tracks = data[(data['cluster'] == cluster_id) & (data['language'] == track_language)]
    cluster_item_matrix = user_item_matrix.reindex(cluster_tracks['track_id']).fillna(0)

    # Find KNN recommendations
    distances, indices = knn.kneighbors([user_item_matrix.loc[track_id]], n_neighbors=num_recommendations + 1)
    recommended_track_indices = indices.flatten()[1:]

    # Next recommended tracks are fetched excluding the input
    recommendations = data[data['track_id'].isin(user_item_matrix.iloc[recommended_track_indices].index)].copy()

    # Next is to get the detailed recommendations
    recommendations = recommendations[recommendations['language'] == track_language]

    if sort_by == "popularity":
        recommendations = recommendations.sort_values(by='popularity', ascending=False)
    elif sort_by == "Release Year":
        recommendations = recommendations.sort_values(by='year', ascending=False)

    # recommended track's duration are formated
    recommendations['duration_minutes'] = recommendations['duration_ms'].apply(duration_minutes)

    return recommendations[['track_name', 'artist_name', 'album_name', 'duration_minutes', 'track_url', 'artwork_url']]


# Start of Web App Creation
# Page configuration is set up
st.set_page_config(page_title="Spotify Music Recommender", layout="wide")

# Web App title
st.title("Spotify Music Recommender")

# Next is to set up a search input for the user
input_query = st.text_input("Enter a track name or artist name:")
sort_by = st.selectbox("sort By:", ["None", "Popularity", "Release Year"])

selected_album = st. session_state.get("selected_album", None)

# Clickable button to get recommendations for user
if st.button("Find Recommendations"):

    if input_query:
        recommendations = recommend_tracks(input_query, data, knn, sort_by=sort_by)

        if isinstance(recommendations, str):
            st.error(recommendations)   # Show error if no match found
        else:
            st.subheader(f"Top {len(recommendations)} Recommendations for: {input_query}")

            for _, row in recommendations.iterrows():
                # Display track information
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.image(row['artwork_url'], width=150)  # Show artwork

                with col2:
                    st.markdown(f"**{row['track_name']}** - {row['artist_name']}")

                    # Added a feature that allows users view all the songs in a recommended songs album
                    if st.button(f"{row['album_name']}", key=row['album_name']):
                        selected_album = row['album_name']
                        st.session_state["selected_album"] = selected_album

                    st.markdown(f"Duration: {row['duration_minutes']}")
                    st.markdown(f"[Listen on spotify]({row['track_url']})")

                st.markdown("----")  # Separator for readability

# This is the code that shows all songs from the selected album above
if selected_album:
    st.subheader(f"All songs from {selected_album}")

    album_tracks = data[data['album_name'] == selected_album]
    for _, track in album_tracks.iterrows():
        st.markdown(f"**{track['track_name']}** - {track['artist_name']}")

# Added basic Visualization at the bottom of web app
# Couldn't really think of any other visualizations to add, dataset was pretty limited in this sense
with st.expander("Show Reports and Visualizations"):
    st.subheader("Popular Genres in Recommendations")
    genre_counts = data['language'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax, palette="coolwarm")
    ax.set_title("Most Common Languages in Recommended Tracks")
    ax.set_ylabel("Number of Tracks")
    st.pyplot(fig)

    st.subheader("Popularity Trends over Time")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x='year', y='popularity', ax=ax)
    ax.set_title("Popularity of Tracks Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Popularity Score")
    st.pyplot(fig)