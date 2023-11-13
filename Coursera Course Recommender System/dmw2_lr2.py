import json
import gzip
import sqlite3
import itertools
import numpy as np
import pandas as pd

def read_and_transform(courses, reviews):
    """
    Return the datasets from Coursera as a pandas DataFrame: 
        df_coursedb:  working file for exploratory data analysis
    """
    print('reading the datasets needed...')

    courses = pd.read_csv(courses)
    reviews = pd.read_csv(reviews)
    coursedb = courses.merge(reviews, on='course_id')
    
    return coursedb

def write_to_db(db, tbl_coursedb, df_coursedb):
    """
    Write the pandas Dataframes needed into an SQLite database
    """
    print('writing data into an SQLite database...')
    conn = sqlite3.connect(db)
    df_coursedb.to_sql(tbl_coursedb, con=conn, index=False, if_exists='replace')  

def prepare_data(courses, reviews, db, tbl_coursedb):
    """
    Extract and load datasets from Coursera as pandas DataFrame, then write these into an SQLite database
    """
    df_coursedb = read_and_transform(courses, reviews)
    write_to_db(db, tbl_coursedb, df_coursedb)
    print('data prepared and ready for analysis.')
    
if __name__ == '__main__':
    prepare_data()
    

    
    

    

