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

###############
# REQUEST 2.1 #
###############

def get_users_dataset_base(original_dataset):
    # empty user dataframe
    users_dataset_base = pd.DataFrame(original_dataset.user_id.unique(), columns = ["user_id"])

    '''
        The first feature we want to build is:
        **Favorite genre (i.e., the genre on which the user spent the most time)**
        We can exploit the `explode` method of Pandas to get from each entry a new entry for each genre in the list:
    '''

    # create exploded dataframe
    exploded_dataset = original_dataset.explode("genres")[["user_id", "genres", "duration"]]

    # add together all values with same user_id and same genre
    exploded_dataset = exploded_dataset.groupby(["user_id", "genres"]).sum()

    # get most watched genre
    exploded_dataset = exploded_dataset.groupby("user_id").idxmax()
    exploded_dataset.duration = exploded_dataset.duration.apply(lambda row: row[1])

    # rename column
    exploded_dataset = exploded_dataset.rename(columns={'duration': 'favorite_genre'})

    # merge into user dataset
    users_dataset_base = users_dataset_base.merge(exploded_dataset, left_on = "user_id", right_index = True)



    '''
        The second feature to extract is:
        **Average click duration**
        This is very fast as we have just to calculate the mean of the `duration` columns of every user:
    '''

    # get mean
    mean_dataset = original_dataset.groupby("user_id").duration.mean().reset_index()

    # rename column
    mean_dataset = mean_dataset.rename(columns={'duration': 'mean_duration'})

    # merge into user dataset
    users_dataset_base = users_dataset_base.merge(mean_dataset)



    '''
        The third feature feature is:
        **Time of the day (Morning/Afternoon/Night) when the user spends the most time on the platform (the time spent is tracked through the duration of the clicks)**

        We will use as delimitations between the three time parts:
        - Morning: 6 to 14
        - Afternoon: 14 to 20
        - Night: 20 to 6
    '''

    # create working dataframe substituting the whole datetime with hour
    hours = original_dataset[["user_id", "datetime", "duration"]].copy()
    hours.datetime = hours.datetime.dt.hour

    # cluster hours into three time periods
    hours.datetime = hours.datetime.apply(lambda x: 'morning' if 6 <= x < 14 else 'afternoon' if 14 <= x < 20 else 'night')

    # get most active time period for each user
    hours = hours.groupby(["user_id", "datetime"]).sum().groupby("user_id").idxmax()
    hours.duration = hours.duration.apply(lambda row: row[1])

    # rename column
    hours = hours.rename(columns={'duration': 'favorite_time_day'})

    # merge into user dataset
    users_dataset_base = users_dataset_base.merge(hours, left_on = "user_id", right_index = True)



    '''
        The fourth feature is:
        **Is the user an old movie lover, or is he into more recent stuff (content released after 2010)?**
    '''

    # convert all datetimes to only the year 
    years = original_dataset[["user_id", "release_date"]].copy()
    years.release_date = years.release_date.dt.year

    # check if film is old or not
    years["new"] = years.release_date > 2010
    years["old"] = years.release_date <= 2010

    # count for each user how many have release year > 2010 and how many not, use boolean value as new column
    user_oldnew = years.groupby("user_id").old.sum() > years.groupby("user_id").new.sum()

    # rename column
    user_oldnew.name = "is_oldmovies_lover"

    # merge into user dataset
    users_dataset_base = users_dataset_base.merge(user_oldnew, left_on = "user_id", right_index = True)



    '''
        The fifth feature is:
        **Average time spent a day by the user (considering only the days he logs in)**
    '''

    # we can just take the mean_duration column and divide it by the number of different days the same user has in the dataset.

    # we count how many different days every user has logged in
    days = original_dataset[["user_id", "datetime"]].copy()
    days["datetime"] = days["datetime"].dt.floor('D')
    days = days.groupby("user_id").datetime.nunique()

    # we count the total duration per user
    sums = original_dataset.groupby("user_id").duration.sum()

    # divide total duration by number of days
    average_per_day = sums / days

    # rename column
    average_per_day.name = "duration_per_day"

    # merge into user dataset
    users_dataset_base = users_dataset_base.merge(average_per_day, left_on = "user_id", right_index = True)

    return users_dataset_base

def get_users_dataset_extended(original_dataset):
    # create empty user dataframe
    users_dataset_extended = pd.DataFrame(original_dataset.user_id.unique(), columns = ["user_id"])

    # watchtime feature
    n_movies = original_dataset.groupby("user_id").duration.sum()
    n_movies.name = "total watchtime"

    users_dataset_extended = users_dataset_extended.merge(n_movies, left_on = "user_id", right_index = True)


    # number of iterations feature
    n_iterations = original_dataset.groupby("user_id").size()
    n_iterations.name = "iterations"

    users_dataset_extended = users_dataset_extended.merge(n_iterations, left_on = "user_id", right_index = True)


    # number of active days feature
    days_active = original_dataset[["user_id", "datetime"]].copy()
    days_active["datetime"] = days_active["datetime"].dt.floor('D')
    days_active = days_active.groupby("user_id").datetime.nunique()
    days_active.name = "active_days"

    users_dataset_extended = users_dataset_extended.merge(days_active, left_on = "user_id", right_index = True)


    # number of unique movies feature
    n_movies = original_dataset.groupby("user_id").movie_id.nunique()
    n_movies.name = "n_movies"

    users_dataset_extended = users_dataset_extended.merge(n_movies, left_on = "user_id", right_index = True)


    # highest number of rewatches feature
    rewatch = original_dataset.groupby(["user_id", "movie_id"]).size().groupby("user_id").max()
    rewatch.name = "max_rewatches"

    users_dataset_extended = users_dataset_extended.merge(rewatch, left_on = "user_id", right_index = True)


    # favorite day of the week feature
    favourite_day = original_dataset[["user_id", "datetime"]].copy()
    favourite_day.datetime = favourite_day.datetime.dt.dayofweek
    favourite_day = favourite_day.groupby(["user_id", "datetime"]).size().groupby("user_id").idxmax().apply(lambda x: x[1])
    favourite_day.name = "favorite_day"

    users_dataset_extended = users_dataset_extended.merge(favourite_day, left_on = "user_id", right_index = True)


    # favorite month of the year feature
    favourite_month = original_dataset[["user_id", "datetime"]].copy()
    favourite_month.datetime = favourite_month.datetime.dt.month
    favourite_month = favourite_month.groupby(["user_id", "datetime"]).size().groupby("user_id").idxmax().apply(lambda x: x[1])
    favourite_month.name = "favorite_month"

    users_dataset_extended = users_dataset_extended.merge(favourite_month, left_on = "user_id", right_index = True)


    # most active year feature
    most_active_year = original_dataset[["user_id", "datetime"]].copy()
    most_active_year.datetime = most_active_year.datetime.dt.year
    most_active_year = most_active_year.groupby(["user_id", "datetime"]).size().groupby("user_id").idxmax().apply(lambda x: x[1])
    most_active_year.name = "most_active_year"

    users_dataset_extended = users_dataset_extended.merge(most_active_year, left_on = "user_id", right_index = True)


    # weekend person feature
    def weekend_function(row):
        row = row.droplevel(0)
        row = row.reindex(range(7), fill_value=0)
        return (row[5] + row[6]) > (row[0] + row[1] + row[2] + row[3] + row[4])

    weekend = original_dataset[["user_id", "datetime"]].copy()
    weekend.datetime = weekend.datetime.dt.dayofweek
    weekend = weekend.groupby(["user_id", "datetime"]).size()
    weekend = weekend.groupby("user_id").apply(weekend_function)
    weekend.name = "is_weekend_user"

    users_dataset_extended = users_dataset_extended.merge(weekend, left_on = "user_id", right_index = True)


    # longest iteration feature
    longest_iteration = original_dataset.groupby("user_id").duration.max()
    longest_iteration.name = "longest_iteration"

    users_dataset_extended = users_dataset_extended.merge(longest_iteration, left_on = "user_id", right_index = True)

    return users_dataset_extended

###############
# REQUEST 2.2 #
###############

def cramers_v(confusion_matrix):
    # Calculate the chi-squared statistic from the contingency matrix
    chi2 = chi2_contingency(confusion_matrix)[0]

    # Calculate the total number of observations (total number of rows) in the contingency matrix
    n = confusion_matrix.sum().sum()

    # Calculate phi-squared which is a measure of association strength
    phi2 = chi2 / n

    # Calculate corrected phi-squared 
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))

    # Calculate the number of rows and columns after correction
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    # Calculate and return Cramér's V with bias correction
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def association_matrix(data_frame):
    # Get the column names (variables) of the DataFrame
    columns = data_frame.columns

    # Create an empty DataFrame to store the association matrix
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    # Iterate over pairs of columns and compute Cramér's V for each pair
    for col1, col2 in combinations(columns, 2):
        # Create a contingency table for the pair of variables
        contingency_table = pd.crosstab(data_frame[col1], data_frame[col2])

        # Calculate Cramér's V with bias correction using the previously defined function
        matrix.loc[col1, col2] = cramers_v(contingency_table)

        # Set the symmetric value on the association matrix
        matrix.loc[col2, col1] = matrix.loc[col1, col2]

    # Set the diagonal elements to 1.0
    np.fill_diagonal(matrix.values, 1.0)
    
    # Return the final association matrix
    return matrix

###############
# REQUEST 2.3 #
###############

def k_means(data, n_clusters, max_iterations):
    # create local spark session
    sc = SparkContext("local", "KMeans")

    # convert data in pyspark format
    data = sc.parallelize(data)

    # initiazation
    # the initial cluster representatives among all points at random, without replacement
    representatives = data.takeSample(False, n_clusters)

    # we execute a fixed number of iterations, if we don't find convergence first
    for _ in range(max_iterations):

        # the first map operation of the algorithm finds which elements belong to which cluster
        # by calculating the distance of each element with every centroid and taking the one that
        # minimizes the euclidean distance
        # the output element is composed as follows:
        # [ cluster_id, (points coordinates, accumulator) ]
        # the accumulator is a variable that will help us keep track of how many points are inside a
        # cluster, it will be necessary to calculate the mean later
        closest = data.map(lambda row: (np.argmin([np.linalg.norm(row - r) for r in representatives]), (row, 1)))

        # the first reduce operation of the algorithm calculate the sum of all alements that belong to a
        # certain cluster. this will group values by the first element (the cluster they belong) and sum
        # both elements of the tuple
        sums = closest.reduceByKey(lambda row1, row2: (row1[0] + row2[0], row1[1] + row2[1]))

        # finally we execute a new map operation to calculate the mean, obtaining the new representatives
        # for every cluster it takes the corresponding tuple containing the sum of all the points of the
        # cluster and the number of points and execute a division, obtaining the mean
        old_representatives = representatives
        representatives = sums.map(lambda cluster_sum: cluster_sum[1][0] / cluster_sum[1][1]).collect()
        
        # if all the old and the new representatives are the same we have found convergence and won't
        # need additional iterations
        if (np.equal(old_representatives, representatives).all()):
            print("Convergence found!")
            break

    clustered_points = []
    for point in closest.collect():
        # [cluster_id | coordinates]
        clustered_points.append((point[0], point[1][0]))

    # stop spark session
    sc.stop()

    return representatives, clustered_points

def k_means_plus_plus(data, n_clusters, max_iterations):
    # initiazation

    # choose the first centroid randomly
    representatives = [data[np.random.randint(data.shape[0])]]

    for _ in range(1, n_clusters):
        # calculate squared distances from each point to the nearest centroid
        dists = np.array([min([np.linalg.norm(d - r)**2 for r in representatives]) for d in data])

        # array of probability for each point to be taken
        # if a point is already a representative the probanility will be zero
        # and it will not be extracted again
        probs = dists / dists.sum()

        # random extraction
        new = data[np.random.choice(len(data), p = probs)]

        # add as new cluster representative
        representatives.append(new)
    
    # create local spark session
    sc = SparkContext.getOrCreate()

    # convert data in pyspark format
    data = sc.parallelize(data)
    
    # the rest of the algorithm is the same as before
    for _ in range(max_iterations):

        closest = data.map(lambda row: (np.argmin([np.linalg.norm(row - r) for r in representatives]), (row, 1)))

        sums = closest.reduceByKey(lambda row1, row2: (row1[0] + row2[0], row1[1] + row2[1]))

        old_representatives = representatives
        representatives = sums.map(lambda cluster_sum: cluster_sum[1][0] / cluster_sum[1][1]).collect()
        
        if (np.equal(old_representatives, representatives).all()):
            print("Convergence found!")
            break

    clustered_points = []
    for point in closest.collect():
        # [cluster_id | coordinates]
        clustered_points.append((point[0], point[1][0]))

    # stop spark session
    sc.stop()

    return representatives, clustered_points