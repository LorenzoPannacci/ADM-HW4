# import libraries and config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark import SparkContext
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from prince import FAMD, MCA, PCA
from yellowbrick.cluster import KElbowVisualizer
from operator import itemgetter
from itertools import combinations
from collections import defaultdict, Counter

import random
random.seed(42)

import warnings
warnings.filterwarnings("ignore")

# functions

def data_gathering(original_dataset):

    # create a new dataframe with the information I need
    users = original_dataset[["genres", "user_id","title"]]

    # group user and movie and count the click
    user_movie_counts = users.groupby(['user_id', 'title']).size().reset_index(name='click_count')

    # sort them
    user_movie_counts = user_movie_counts.sort_values(by=['user_id', 'click_count'], ascending=[True, False])

    # take the top 10 movies for each user
    user_movie_counts = user_movie_counts.groupby('user_id').head(10)

    # return a new dataframe with the top 10 movies for each user
    # we keep as informations the user, the title and the genre of the movie as well as the click count
    # other columns are not requested and will be removed

    result = pd.merge(user_movie_counts, users[['user_id', 'title', 'genres']], on=['user_id', 'title'], how='left')

    # as list are unhashable we convert genres back to string
    result['genres'] = result['genres'].apply(lambda x: ', '.join(map(str, x)))

    # finally we have to remove duplicates
    # this is needed as the previous operations gave us the right answer, but
    # every row is repeated for each click count it has, so for example if a
    # movie has 9 clicks, it creates 9 times that rows
    result.drop_duplicates(inplace = True)

    return result

def get_user_genres_df(original_dataset):
    # 0)
    # we create a 2 columns dataframe: one with the users_id and one with the genres, each row will only have one genre
    # this will be needed for creating a dictionary with users as keys and genres as values

    genres_with_user = original_dataset[["user_id", "genres"]]
    genres_with_user = genres_with_user.explode('genres').reset_index(drop = True)
    genres_with_user

    # create a dictionary where we have as keys the user_id and as values all genres related to it

    user_genres_dict = {}

    # iterate over dataframe
    for index, row in genres_with_user.iterrows():
        user_id = row['user_id']
        genre = row['genres']

        # if user_id already exist, append 
        if user_id in user_genres_dict:
            user_genres_dict[user_id].append(genre)

        # otherwise, create the key
        else:
            user_genres_dict[user_id] = [genre]

    # then we convert into a dataframe (it will be necessary later)
    user_genres_df = pd.DataFrame(list(user_genres_dict.items()), columns=['user_id', 'genres'])
    return user_genres_df

def get_unique_genres(original_dataset):
    # retrieve all unique film genres
    unique_genres = set()
    original_dataset["genres"].apply(lambda row: [unique_genres.add(value) for value in row])
    unique_genres = sorted(list(unique_genres))

    return unique_genres


def get_users_genres_matrix(user_genres_df, unique_genres):
    # 1) create the matrix 

    # create my matrix: columns are genres and the index is user_id
    users_genres_matrix = pd.DataFrame(0, index=user_genres_df["user_id"].unique(), columns = unique_genres)

    # put 1 if the user has that genre, put 0 if not
    for _, row in user_genres_df.iterrows():
        user_id = row["user_id"]
        genres = row["genres"]

        for genre in genres:
            if genre in unique_genres:
                users_genres_matrix.loc[user_id, genre] = 1

    return users_genres_matrix

def generate_hash(n_hashes):
    # we create the hash function
    # we give a number of choice and create that number of functions, the coefficient x and y are different every time
    # the equation is fixed but the parameters are changede randomly

    hash_functions = []

    for _ in range(n_hashes):
        x = random.randint(1, 100)
        y = random.randint(0, 100)
        hash_functions.append((x, y))

    return hash_functions

def calculate_signature(user_preferences, n_hashes, n_genres, list_functions):
    signature = [float('inf')] * n_hashes

    # for every element of my matrix (which is actually a dataframe)
    # for every genre index and the binary encoding (0, 1)
    for index, present in enumerate(user_preferences):

        # for each genre execute the hash functions
        for k, hash_function in enumerate(list_functions):
            x, y = hash_function
            hash_value = (x * index * present + y) % n_genres
            # retain only the minimun
            signature[k] = min(signature[k], hash_value)

    # so here we will have the minimum
    return signature

def get_signatures(users_genres_matrix, n_hashes, n_genres, list_functions):
    # dictionary that will store the signatures for each user
    signatures = {}

    #I calculare the signature for each user in the matrix (which is a dataframe actually)
    for user_id, user_preferences in users_genres_matrix.iterrows():
        signature = calculate_signature(user_preferences, n_hashes, n_genres, list_functions)
        signatures[user_id] = signature
    
    return signatures

def bucketing(signatures, n_hashes, n_bands):
    # number of rows per band
    n_rows = n_hashes // n_bands

    # dictionary that will store the buckets
    buckets = defaultdict(list)

    # for each user:
    for user_id, signature in signatures.items():
        for band_index in range(n_bands):
            # calculates the start and end indices for the current band.
            start = band_index * n_rows
            end = (band_index + 1) * n_rows

            # extract the band from the minhash signature and convert it to a tuple
            band = tuple(signature[start:end])

            # append the user to the corresponding bucket based on the value.
            buckets[band].append(user_id)

    return buckets

def find_most_similar_users(buckets, user_id, n_users=10):
    # the idea is that given a user we count occurrences in the same buckets of the other users
    # if our input user and the other users appears into a lot of common buckets, it is very likely that they have same interests
    # the output is a list sorted in descending order of occurrences
    
    # make a list with all users into the buckets in which the input user_id appears
    user_buckets = [bucket for bucket in buckets.values() if user_id in bucket]
    all_users = [user for bucket in user_buckets for user in bucket if user != user_id]

    # count occurrences of users
    user_occurrences = Counter(all_users)

    # sort
    sorted_user_occurrences = user_occurrences.most_common(n_users)
    
    return sorted_user_occurrences

def calculate_jaccard_similarity(user1_genres, user2_genres):
    #jaccard similarity
    intersection = len(set(user1_genres) & set(user2_genres))
    union = len(set(user1_genres).union(user2_genres))
    return intersection / union

def select_by_similarity(user_genres_df, most_similar_users, user_id_input):
    # given the users we received from the previous filtering we compute on them
    # jaccard similarity and sort them
    # then we take the 2 users with the highest score
    
    input_user_genres = user_genres_df[user_genres_df['user_id'] == user_id_input]['genres'].iloc[0]

    # store the similarity result
    similarities = []

    for similar_user_id, _ in most_similar_users:
        similar_user_genres = user_genres_df[user_genres_df['user_id'] == similar_user_id]['genres'].iloc[0]
        
        jaccard_similarity = calculate_jaccard_similarity(input_user_genres, similar_user_genres)
        
        similarities.append({'user_id': similar_user_id, 'similarity': jaccard_similarity})

    similarities_df = pd.DataFrame(similarities)
    similarities_df = similarities_df.sort_values("similarity", ascending= False)

    return similarities_df

def get_common_movies(top_2_users, top10_movies, return_df = False):
    user_1 = top_2_users.iloc[0]
    user_2 = top_2_users.iloc[1]

    # extract common movies

    # filter data for each user
    user1_movies = top10_movies[top10_movies['user_id'] == user_1]
    user2_movies = top10_movies[top10_movies['user_id'] == user_2]

    # merge the dataframes such that I have all the movies together
    merged_users = pd.concat([user1_movies, user2_movies])

    # we suggest all movies that are in common and we order them by clicks count
    user_groups = merged_users.groupby('user_id')['title'].apply(set)

    # find the common titles between the two users
    common_titles = set.intersection(user_groups.iloc[0], user_groups.iloc[1])

    common_df = merged_users[merged_users['title'].isin(common_titles)].copy()

    # count clicks for each film
    common_df['sum_click_count'] = common_df.groupby('title')['click_count'].transform('sum')

    # sort by the sum of cliks
    sorted_common_df = common_df.sort_values(by='sum_click_count', ascending=False)

    # films in common between the two users, drop dublicates
    sorted_common_df = sorted_common_df.drop_duplicates(subset='title')
    sorted_common_df = sorted_common_df[["title","sum_click_count"]]
    
    if return_df:
        return sorted_common_df, merged_users
    else:
        return sorted_common_df

def get_recommendations(top_2_users, top10_movies):
    # if the common films are 5, we are done
    # if they are less than 5, we suggest the remaining ones from the user by greatest similarity
    # if the user with greatest similarity has no more other films we will pass to the second user
    # EX: if the common films are 3, I will suggest only 2 films from the user with greatest similarity / the second user

    # check if it has more than 5 rows, if it is the case, suggest the first 5
    # if it is not the case, use following code

    common_movies, merged_users = get_common_movies(top_2_users, top10_movies, return_df = True)

    value = True
    key = common_movies.shape[0]

    if common_movies.shape[0] >= 5:
        value = True
        print(common_movies['title'].head(5).to_list())
    else:
        value = False

    # if there are less than 5 films, then suggest others in base of the user with the most similarity:

    # we create a new dataframe with the films of the two users, but without the one(s) in common
    final = 5
    sub_key = final - key

    common_titles = set(merged_users['title'].value_counts()[merged_users['title'].value_counts() > 1].index)

    # remove rows with common titles
    filtered_df = merged_users[merged_users['title'].isin(common_titles) == False]

    new_df = filtered_df.head(sub_key)

    # first I have the common films, then we have the only-one-user films
    recomended_movies = list(pd.concat([common_movies, new_df])["title"])

    return recomended_movies