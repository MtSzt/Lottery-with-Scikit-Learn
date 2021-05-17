import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 

def predict_general(dt, next_count, ball_col):
    
    X_dates = dt.iloc[:, 0].values.reshape(-1,1)
    y_ball1_numbers = dt.iloc[:, ball_col].values.reshape(-1,1)
    np.ravel(y_ball1_numbers)
    to_predict_x = next_count
    to_predict_x = np.array(to_predict_x).reshape(-1,1)

    regsr = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regsr.fit(X_dates,np.ravel(y_ball1_numbers))

    predicted_y = regsr.predict(to_predict_x)
    
    return (predicted_y.round) 

def predict_ball1(dt, next_count):
    #
    X_dates = dt.iloc[:, 0].values.reshape(-1,1)
    y_ball1_numbers = dt.iloc[:, 2].values.reshape(-1,1)

    to_predict_x = next_count
    to_predict_x = np.array(to_predict_x).reshape(-1,1)

    regsr = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regsr.fit(X_dates,np.ravel(y_ball1_numbers))

    predicted_y = regsr.predict(to_predict_x)
    
    return (predicted_y.round) 

def predict_ball2(dt, next_count, ball1):
    
    df1 = dt.loc[dt['ball-1'] == ball1]

    X_dates = df1.iloc[:, 0].values.reshape(-1,1)
    y_ball1_numbers = df1.iloc[:, 3].values.reshape(-1,1)

    to_predict_x = next_count
    to_predict_x = np.array(to_predict_x).reshape(-1,1)

    regsr = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regsr.fit(X_dates,np.ravel(y_ball1_numbers))

    predicted_y = regsr.predict(to_predict_x)
    
    return (predicted_y.round) 

def predict_ball3(dt, next_count, ball1, ball2):
    
    df1 = dt.loc[(dt['ball-1'] == ball1) & ((dt['ball-2'] == ball2))]

    X_dates = df1.iloc[:, 0].values.reshape(-1,1)
    y_ball1_numbers = df1.iloc[:, 4].values.reshape(-1,1)

    to_predict_x = next_count
    to_predict_x = np.array(to_predict_x).reshape(-1,1)

    regsr = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regsr.fit(X_dates,np.ravel(y_ball1_numbers))

    predicted_y = regsr.predict(to_predict_x)
    
    return (predicted_y.round)

def predict_ball4(dt, next_count, ball1, ball2, ball3):
    
    df1 = dt.loc[(dt['ball-1'] == ball1) & ((dt['ball-2'] == ball2)) & ((dt['ball-3'] == ball3))]

    X_dates = df1.iloc[:, 0].values.reshape(-1,1)
    y_ball1_numbers = df1.iloc[:, 5].values.reshape(-1,1)

    to_predict_x = next_count
    to_predict_x = np.array(to_predict_x).reshape(-1,1)

    regsr = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regsr.fit(X_dates,np.ravel(y_ball1_numbers))

    predicted_y = regsr.predict(to_predict_x)
    
    return (predicted_y.round)

def predict_ball5(dt, next_count, ball1, ball2, ball3, ball4):
    
    df1 = dt.loc[(dt['ball-1'] == ball1) & ((dt['ball-2'] == ball2)) & ((dt['ball-3'] == ball3)) & ((dt['ball-4'] == ball4))]

    X_dates = df1.iloc[:, 0].values.reshape(-1,1)
    y_ball1_numbers = df1.iloc[:, 6].values.reshape(-1,1)

    to_predict_x = next_count
    to_predict_x = np.array(to_predict_x).reshape(-1,1)

    regsr = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regsr.fit(X_dates,np.ravel(y_ball1_numbers))

    predicted_y = regsr.predict(to_predict_x)
    
    return (predicted_y.round)

def predict_ball6(dt, next_count, ball1, ball2, ball3, ball4, ball5):
    
    df1 = dt.loc[(dt['ball-1'] == ball1) & ((dt['ball-2'] == ball2)) & ((dt['ball-3'] == ball3)) & ((dt['ball-4'] == ball4)) & ((dt['ball-5'] == ball5))]

    X_dates = df1.iloc[:, 0].values.reshape(-1,1)
    y_ball1_numbers = df1.iloc[:, 7].values.reshape(-1,1)

    to_predict_x = next_count
    to_predict_x = np.array(to_predict_x).reshape(-1,1)

    regsr = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regsr.fit(X_dates,np.ravel(y_ball1_numbers))

    predicted_y = regsr.predict(to_predict_x)
    
    return (predicted_y.round)
