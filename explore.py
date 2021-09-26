import numpy as np
import pandas as pd
#For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
#For clustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#The following function will create a relplot of logerror against each numeric variable in the provided dataset
def get_logerror_plots(df, hue = None, col = None):
    #Create a list of all numeric columns names
    num_cols = df.select_dtypes('number').columns

    #Create the charts, but skip the logerror column name
    for col in num_cols:
        if col == 'logerror':
            continue
        else:
            plt.figure(figsize=(16, 8))
            sns.relplot(x = col, y = 'logerror', hue = hue, col = col, data = df)
            plt.show()

#The following function will create three different groups of clusters and attach them to the train, validate, and test sets.
#It takes in the train, validate, test data sets and a list of dictionaries containg the required info for each cluster group.
def get_clusters(train, validate, test, clusters):
    #Loop through each group of cluster features, and create the clusters
    for cluster in clusters:
        #Get the values for each groups columns
        train_cluster = train[cluster['features']].copy()
        validate_cluster = validate[cluster['features']].copy()
        test_cluster = test[cluster['features']].copy()

        #Now scale the data. Fit on train_cluster only
        scaler = MinMaxScaler()

        scaler.fit(train_cluster)

        train_cluster = scaler.transform(train_cluster)
        validate_cluster = scaler.transform(validate_cluster)
        test_cluster = scaler.transform(test_cluster)

        #Create the Kmeans model
        kmeans = KMeans(n_clusters = cluster['k'], random_state = 123)

        #Fit the model on train_cluster
        kmeans.fit(train_cluster)

        #Make predictions and assign values to original train, validate, and test data sets
        train[cluster['name']] = kmeans.predict(train_cluster)
        validate[cluster['name']] = kmeans.predict(validate_cluster)
        test[cluster['name']] = kmeans.predict(test_cluster)

    #return the new dataframes
    return train, validate, test