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

def iterative_call(student_score, marks_list, solved_subproblems = {}):
    # base case, if the marks list has only one element we are forced to take it
    if len(marks_list) == 1:
        return marks_list[0]
    
    # if we have already solved this scenario we can take the result without
    # computing it again another time
    elif (student_score, marks_list) in solved_subproblems:
        return solved_subproblems[(student_score, marks_list)]
    
    # otherwise is a new scenario, we have to compute it by checking every possible path
    else:
        # we create a list of tuples, each tuple is the mark list if we chose to take a certain mark
        new_marks_lists = [tuple(mark + (student_score - marks_list[i]) for mark in marks_list[:i] + marks_list[i+1:]) for i in range(len(marks_list))]

        # we create a list of scores, that are the output of each new subproblem
        new_scores = [iterative_call(marks_list[i], new_marks_lists[i], solved_subproblems) for i in range(len(marks_list))]

        # we get the mark among all subproblems
        best_score = max(new_scores)
        
        # update solved subproblems dictionary
        solved_subproblems[(student_score, marks_list)] = best_score

        return best_score

def algorithmic_question_v1(input_string):
    input_list = input_string.split("\n")

    # get student score from input string
    original_student_score = int(input_list[0])

    # get marks list from input string
    original_marks_list = tuple(map(int, input_list[1].split(" ")))

    best_score = iterative_call(original_student_score, original_marks_list)

    print(best_score)

def algorithmic_question_v2(input_string):
    input_list = input_string.split("\n")

    # get student score from input string
    student_score = int(input_list[0])

    # get marks list from input string
    marks_list = list(map(int, input_list[1].split(" ")))

    # sort values
    marks_list.sort()

    # treat even/odd case
    result = 0 if len(marks_list) % 2 == 1 else student_score

    # sum value above median and substract values below median
    result += sum(marks_list[len(marks_list) // 2 :]) - sum(marks_list[: len(marks_list) // 2])
    
    print(result)