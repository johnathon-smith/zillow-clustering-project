import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.metrics import mean_squared_error, explained_variance_score

#The following function will take in train, validate, and test, create dummies, and then return the new datasets
def get_dummies(train, validate, test):
    #First, set the int categoricals to dtype 'object'
    train[['age_location_cluster', 'size_cluster', 'value_cluster']] = train[['age_location_cluster', 'size_cluster', 'value_cluster']].astype('object')
    validate[['age_location_cluster', 'size_cluster', 'value_cluster']] = validate[['age_location_cluster', 'size_cluster', 'value_cluster']].astype('object')
    test[['age_location_cluster', 'size_cluster', 'value_cluster']] = test[['age_location_cluster', 'size_cluster', 'value_cluster']].astype('object')

    #Get cols to create dummies for
    cat_cols = ['county', 'age_location_cluster', 'size_cluster', 'value_cluster']

    df_dummies = pd.get_dummies(train[cat_cols], dummy_na=False, drop_first=True)
    train = pd.concat([train, df_dummies], axis = 1).drop(columns = cat_cols)

    df_dummies = pd.get_dummies(validate[cat_cols], dummy_na=False, drop_first=True)
    validate = pd.concat([validate, df_dummies], axis = 1).drop(columns = cat_cols)

    df_dummies = pd.get_dummies(test[cat_cols], dummy_na=False, drop_first=True)
    test = pd.concat([test, df_dummies], axis = 1).drop(columns = cat_cols)

    return train, validate, test

#The following function will scale the X data sets using the MinMaxScaler()
def scale_data(X_train, X_validate, X_test):
    #Create the scaler
    scaler = MinMaxScaler()

    #Fit the scaler on X_train
    scaler.fit(X_train)

    #Transform the data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validate_scaled, X_test_scaled

#The following function will create a df to store all of the metric values for each model I create.
#This allows for easy evaluation.
def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE: mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE: mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)

#The following function will get the baseline and print the RMSE.
#It also transforms the y datasets into dataframes
def get_baseline(y_train, y_validate, y_test, metric_df):
    #Change y_train and y_validate to be data frames so we can store the baseline values in them
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    #Calculate baseline based on mean
    baseline_mean_pred = y_train.logerror.mean()
    y_train['baseline_mean_pred'] = baseline_mean_pred
    y_validate['baseline_mean_pred'] = baseline_mean_pred
    y_test['baseline_mean_pred'] = baseline_mean_pred

    #Calculate RMSE based on mean
    train_RMSE = mean_squared_error(y_train.logerror, y_train['baseline_mean_pred']) ** .5
    validate_RMSE = mean_squared_error(y_validate.logerror, y_validate['baseline_mean_pred']) ** .5

    #Print RMSE
    print("RMSE using Mean\nTrain/In-Sample: ", round(train_RMSE, 4), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 4),
        "\n")

    metric_df = make_metric_df(y_validate.logerror, y_validate['baseline_mean_pred'], 'validate_baseline_mean', metric_df)

    return y_train, y_validate, y_test, metric_df

#The following function will create the OLS model, print the RMSE, and add it to the metric df
def get_ols_model(X_train_scaled, X_validate_scaled, y_train, y_validate, metric_df):
    #Create the model
    lm = LinearRegression(normalize = True)

    #Fit the model on scaled data
    lm.fit(X_train_scaled, y_train.logerror)

    #Make predictions
    y_train['lm_preds'] = lm.predict(X_train_scaled)
    y_validate['lm_preds'] = lm.predict(X_validate_scaled)

    #Calculate the RMSE
    train_RMSE = mean_squared_error(y_train.logerror, y_train['lm_preds']) ** .5
    validate_RMSE = mean_squared_error(y_validate.logerror, y_validate['lm_preds']) ** .5

    print("RMSE using OLS\nTrain/In-Sample: ", round(train_RMSE, 4), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 4))

    metric_df = make_metric_df(y_validate.logerror, y_validate['lm_preds'], 'validate_ols', metric_df)

    return lm, metric_df


def get_lars_models(X_train_scaled, X_validate_scaled, y_train, y_validate, metric_df):
    #Create a list to store each model
    lars_models = []

    #Loop through different alpha values. Start with 1.
    for i in range(1, 21):
        #Create the model
        lars = LassoLars(alpha = i)
        
        #Fit the model
        lars.fit(X_train_scaled, y_train.logerror)
        
        #Make predictions
        y_train[f'lars_alpha_{i}'] = lars.predict(X_train_scaled)
        y_validate[f'lars_alpha_{i}'] = lars.predict(X_validate_scaled)
        
        #Calculate RMSE
        train_RMSE = mean_squared_error(y_train.logerror, y_train[f'lars_alpha_{i}']) ** .5
        validate_RMSE = mean_squared_error(y_validate.logerror, y_validate[f'lars_alpha_{i}']) ** .5

        #Add model to list of lars models
        lars_models.append({f'lars_alpha_{i}': lars})
        
        print(f'\nRMSE using LassoLars, alpha = {i}')
        print("Train/In-Sample: ", round(train_RMSE, 4), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 4))

        metric_df = make_metric_df(y_validate.logerror, y_validate[f'lars_alpha_{i}'], f'validate_lars_alpha_{i}', metric_df)

    return lars_models, metric_df

#The following function will create RandomForestRegressor models, print the RMSE, and add them to the metric df
def get_rfr_models():
    #Starting from 2 in order to avoid warnings
    rfr_models = []

    for num in range(10, 16):
        #Now create a new loop that runs through different min_samples_leaf values
        for val in range(15, 21):
            #Instantiate new model
            model = RandomForestRegressor(random_state = 123, max_depth = num, min_samples_leaf = val)

            #Fit the model
            model.fit(X_train_scaled, y_train.logerror)

            #Make predictions
            y_train[f'rfr_depth_{num}_samples_{val}_preds'] = model.predict(X_train_scaled)
            y_validate[f'rfr_depth_{num}_samples_{val}_preds'] = model.predict(X_validate_scaled)
            
            #Calculate RMSE
            train_RMSE = mean_squared_error(y_train.logerror, y_train[f'rfr_depth_{num}_samples_{val}_preds']) ** .5
            validate_RMSE = mean_squared_error(y_validate.logerror, y_validate[f'rfr_depth_{num}_samples_{val}_preds']) ** .5

            #Add model to the list
            rfr_models.append({f'rfr_depth_{num}_samples_{val}': model})

            print(f'\nRMSE for Max Depth = {num}, Min Samples = {val}\n')
            print("Train/In-Sample: ", round(train_RMSE, 4), 
                "\nValidate/Out-of-Sample: ", round(validate_RMSE, 4))
        
            #Add results to metric dataframe
            metric_df = make_metric_df(y_validate.logerror, y_validate[f'rfr_depth_{num}_samples_{val}_preds'], f'validate_rfr_depth_{num}_samples_{val}', metric_df)
        
    return rfr_models, metric_df
