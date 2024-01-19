# %%
"""
# THEORY QUESTIONS
## Q1-Assume that you have a large training dataset. Specify a disadvantage of the k-Nearest Neighbor method when using it during testing. State also your reason about your answer.

When dealing with large training datasets, it is important to understand a major shortcoming of the k-Nearest Neighbor (k-NN) method in terms of testing. The main issue lies in the mathematical complexity of forecasts. Here is a brief explanation from the professor:

Disadvantages: High computational cost of k-NN in testing

The k-NN algorithm that helps us make predictions is very limited when working with large training data. This is a non-parametric method, which means that the prediction is done using the entire set of training data. During testing, the distance between the test data point and all data points in the training set should be measured to find the nearest k-neighbor. As the size of the training dataset increases, the time required for each prediction increases exponentially.

For large training datasets, this computational burden can be large and inefficient, resulting in significantly delayed prediction times and, moreover, the total memory required to store the training dataset can be large, and has made it more difficult.

To overcome this issue, alternative designs with more efficient prediction methods should be explored or resolution reduction techniques should be considered when working with extensive datasets These methods can help to the prediction has performed well and the computational challenges posed by large training datasets have been addressed.
"""

# %%
"""
# Q2
 ## Considering the image below, state an optimal k-value depending on that the algorithm you are using is k-Nearest Neighbor. State also your reason behind the optimal value you preferred.



![Image](q2.png)

## Answer 
* The optimal k value in k nearest neighbor k nn is decided by striking a balance between bias and variance in the model it involves empirical testing cross validation visual inspection and domain knowledge we select k values run the k nn algorithm for different k and assess the model s performance the goal is to avoid overfitting with low k and underfitting with high k cross validation helps find the sweet spot where bias and variance are balanced the optimal k value approximately 11 or 12 is considered optimal because it captures local patterns while reducing sensitivity to noise visual inspection and error analysis show that it s a robust choice domain knowledge may also influence this decision in a nutshell the process involves practical testing finding the right balance and considering domain specific factors to select the optimal k value for k nn

"""

# %%
"""
# Q3
 

![Image](q3.png)


## Assume that you have the following training set of positive (+), negative (-) instances and a single test instance (o) in the image below (Figure 1). Assume also that the Euclidean metric is used for measuring the distance between instances. Finally consider that every nearest neighbor instance affects the final vote equally.
* What is the class appointed to the test instance for K=1? State also reason behind your answer.

> If we select k=1 in the k-Nearest Neighbor (k-NN) algorithm, it can result in a negative(-) prediction. This occurs because the nearest data point, which determines the prediction, has a negative value.



* What is the class appointed to the test instance for K=3? State also reason behind your answer.

>  If we select k=3 in the k-Nearest Neighbor (k-NN) algorithm, the prediction can still be negative if the majority of the nearest neighbors are negative. Specifically, if two out of the three nearest neighbors are negative and one is positive, the prediction will be negative. This is because k-NN takes into account the majority class among the k-nearest neighbors.



* What is the class appointed to the test instance for K=5? State also reason behind your answer.

> When selecting k=5 in the k-Nearest Neighbor (k-NN) algorithm,  If you have 3 positive neighbors and 2 negative neighbors among the 5 nearest data points, the prediction will be positive.
"""

# %%
"""

# 4. Fill the blanks with T (True) or F (False) for the statements below:

> If all instances of the data have the same scale then k-Nearest Neighbor’s perfor- mance increases drastically. (T )

> While k-Nearest Neighbor performs well with a small number of input variables, it’s performance decreases when the number of inputs becomes large. (T )


> k-Nearest Neighbor makes supposes nothing about the functional form of the problem it handles. (T )
"""

# %%
"""
# Linear Regression Q1

![Image](L1.png)


## ANSWER 
 

![Image](L1_Answer.png)


> Max : 8464

> Min : 2025

> Sample Mean : 7569 + 4900 + 8464 + 4489 + 2025 / 5 = 5489.4

> Range : 8464-2025 = 6439

# Answer :  2025-5489 / ( 6439) =  -0.538




"""

# %%
"""
# Linear Regression Q2
Considering the figure below, which of the offsets used in linear regressions least square line fit? Assume that horizontal axis represents independent variable and vertical axis represents dependent variable. State your answer with your proper explanation.
        

![Image](L2.png)

## Answer

The best offset to consider in linear regression using the least squares line fit is the vertical offset. This method aims to minimize the squared vertical (y-axis) differences between observed data points and the predicted values on the regression line.
"""

# %%
"""
# Linear Regression Q4


##  State a valid reason for feature scaling and explain why it is a valid reason with respect to your reasoning.

> Feature scaling is crucial in machine learning to ensure fair and effective model training it s necessary because equal influence features often have different scales without scaling those with larger scales can disproportionately affect the model potentially leading to incorrect predictions convergence scaling speeds up the optimization process when features have varying scales it can take longer for optimization algorithms to find the best model parameters distance metrics distance based algorithms can be affected by feature scales scaling ensures that distances are accurately calculated preventing bias in algorithms like k nearest neighbors in summary feature scaling is essential to create a level playing field for all features ensuring fair and accurate model training while improving algorithm efficiency
"""

# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

# %%
import pandas as pd

# Load the anime data from the 'animes.csv' file
anime_data = pd.read_csv('animes.csv')

# Load the user ratings training dataset with a limit of 300 rows
user_rates_train = pd.read_csv('user_rates_train.csv')

# Load the user ratings test dataset
user_rates_test = pd.read_csv('user_rates_test.csv')

# Print the shape (number of rows and columns) of the 'anime_data' DataFrame
# to check the dimensions of the dataset



# %%
# Merge the 'anime_data' DataFrame and 'user_rates_train' DataFrame using the 'anime_id' column as the key for the merge operation.
# This combines information about the anime from 'anime_data' with the user ratings from 'user_rates_train'.
# The 'suffixes' parameter is used to distinguish columns with the same names in the merged DataFrame.
fulldata = pd.merge(anime_data, user_rates_train, on='anime_id', suffixes= ['', '_user'])

# Rename the columns in the merged DataFrame 'fulldata' to make them more descriptive.
# Specifically, rename 'name' to 'anime_title' and 'rating_user' to 'user_rating'.
fulldata = fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'})

# Display the first few rows of the 'fulldata' DataFrame using the 'head()' method to inspect the merged and renamed data.
fulldata.head()


# %%
# Merge the 'anime_data' DataFrame and 'user_rates_test' DataFrame using the 'anime_id' column as the key for the merge operation.
# This combines information about the anime from 'anime_data' with the user ratings from 'user_rates_test'.
# The 'suffixes' parameter is used to distinguish columns with the same names in the merged DataFrame.
test_fulldata = pd.merge(anime_data, user_rates_test, on='anime_id', suffixes= ['', '_user'])

# Rename the columns in the merged DataFrame 'test_fulldata' to make them more descriptive.
# Specifically, rename 'name' to 'anime_title' and 'rating_user' to 'user_rating'.
test_fulldata = test_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'})

# Display the first few rows of the 'test_fulldata' DataFrame using the 'head()' method to inspect the merged and renamed data.
test_fulldata.head()


# %%
fulldata.isnull().sum()

# %%
"""
### There is no null data 
"""

# %%
# Split the 'Genres' column in the 'fulldata' DataFrame into binary columns using one-hot encoding.
# The 'str.get_dummies' method is applied to the 'Genres' column with ',' as the separator.
# This creates binary columns for each genre, indicating whether an anime belongs to that genre.
df_genres_list = fulldata['Genres'].str.get_dummies(sep=',')

# Display a sample of 10 rows from the 'df_genres_list' DataFrame to show the one-hot encoded genre information.
df_genres_list.sample(10)

# Split the 'Genres' column in the 'test_fulldata' DataFrame into binary columns using one-hot encoding.
# The 'str.get_dummies' method is applied to the 'Genres' column with ',' as the separator.
# This creates binary columns for each genre in the test set, indicating whether an anime belongs to that genre.
test_df_genres_list = test_fulldata['Genres'].str.get_dummies(sep=',')

# Display a sample of 10 rows from the 'test_df_genres_list' DataFrame to show the one-hot encoded genre information for the test set.
test_df_genres_list.sample(10)

# Check the shape (number of rows and columns) of the 'df_genres_list' DataFrame to determine the dimension of the one-hot encoded genre data.



# %%
# Create binary columns for the 'Type' column in the 'fulldata' DataFrame using one-hot encoding.
# The 'pd.get_dummies' function is applied to the 'Type' column.
df_types_list = pd.get_dummies(fulldata[["Type"]])

# Display a sample of 10 rows from the 'df_types_list' DataFrame to show the one-hot encoded information for anime types in the training dataset.
df_types_list.sample(10)

# Create binary columns for the 'Type' column in the 'test_fulldata' DataFrame using one-hot encoding.
# The 'pd.get_dummies' function is applied to the 'Type' column in the test set.
test_df_types_list = pd.get_dummies(test_fulldata[["Type"]])

# Display a sample of 10 rows from the 'test_df_types_list' DataFrame to show the one-hot encoded information for anime types in the test set.
test_df_types_list.sample(10)

# Check the shape (number of rows and columns) of the 'df_types_list' DataFrame to determine the dimension of the one-hot encoded anime type data.


# %%
# Create binary columns for the 'Source' column in the 'fulldata' DataFrame using one-hot encoding.
# The 'pd.get_dummies' function is applied to the 'Source' column.
df_source_list = pd.get_dummies(fulldata[["Source"]])

# Display a sample of 10 rows from the 'df_source_list' DataFrame to show the one-hot encoded information for anime sources in the training dataset.
df_source_list.sample(10)

# Create binary columns for the 'Source' column in the 'test_fulldata' DataFrame using one-hot encoding.
# The 'pd.get_dummies' function is applied to the 'Source' column in the test set.
test_df_source_list = pd.get_dummies(test_fulldata[["Source"]])

# Display a sample of 10 rows from the 'test_df_source_list' DataFrame to show the one-hot encoded information for anime sources in the test set.
test_df_source_list.sample(10)

# Check the shape (number of rows and columns) of the 'df_source_list' DataFrame to determine the dimension of the one-hot encoded anime source data.



# %%
import re

def convert_to_minutes(duration_str):
    """
    Convert a duration string to minutes.

    Args:
        duration_str (str): A string representing a duration in various time units.

    Returns:
        int: The equivalent duration in minutes. Returns -1 if the input is unparseable.
    """
    # Default value for unparseable strings or unknowns
    minutes = -1

    # Define a dictionary to map time units to their conversion factors
    unit_to_minutes = {
        'hr': 60,    # Convert hours to minutes
        'min': 1,    # Minutes remain unchanged
        'sec': 1/60  # Convert seconds to minutes
    }

    # Use regular expressions to extract numeric values and units
    matches = re.findall(r'(\d+)\s*([a-z]+)', duration_str)

    if matches:
        minutes = 0
        for value, unit in matches:
            if unit in unit_to_minutes:
                minutes += int(value) * unit_to_minutes[unit]

    return minutes


# %%
fulldata['Duration'] = fulldata['Duration'].apply(convert_to_minutes)
test_fulldata['Duration'] = test_fulldata['Duration'].apply(convert_to_minutes)

# %%
fulldata.head()

# %%
df_comers = fulldata[['Name','Duration','anime_id',"user_id","rating"]]

test_fulldata_comers = test_fulldata[['Name','Duration','anime_id',"user_id","rating"]]

# %%
df_features = pd.concat([df_comers,df_genres_list, df_types_list,df_source_list], axis = 1).fillna(0)
#df_features = df_features.drop_duplicates()

test_df_features = pd.concat([test_fulldata_comers,test_df_genres_list, test_df_types_list,test_df_source_list], axis = 1).fillna(0)
#df_features = df_features.drop_duplicates()




# %%
df_features.head()


# %%


# %%
features = df_features.drop(['Name'], axis=1)

test_features = test_df_features.drop(["Name"],axis=1)


# %%
from sklearn.preprocessing import MinMaxScaler

# Create a Min-Max Scaler instance to scale the features
min_max_scaler = MinMaxScaler()

# Apply Min-Max scaling to the 'features' DataFrame
scaled_animes = min_max_scaler.fit_transform(features)

# Create a new DataFrame 'scaled_animes' to store the scaled data, maintaining column names
scaled_animes = pd.DataFrame(scaled_animes, columns=features.columns)


# %%
from sklearn.preprocessing import MinMaxScaler

# Create a Min-Max Scaler instance for scaling the test features
min_max_scaler_test = MinMaxScaler()

# Apply Min-Max scaling to the 'test_features' DataFrame
test_scaled_animes = min_max_scaler_test.fit_transform(test_features)

# Create a new DataFrame 'test_scaled_animes' to store the scaled test data while maintaining column names
test_scaled_animes = pd.DataFrame(test_scaled_animes, columns=test_features.columns)


# %%
import numpy as np

def calculate_adjusted_cosine_similarity(vector1, vector2):
    """
    Compute adjusted cosine similarity between two vectors.
    
    Args:
        vector1 (numpy.ndarray): First vector.
        vector2 (numpy.ndarray): Second vector.
    
    Returns:
        float: Adjusted cosine similarity between the two vectors.
    """
    # Calculate the mean of each vector
    mean1 = np.mean(vector1)
    mean2 = np.mean(vector2)

    # Adjust the vectors by subtracting their mean values
    vector1_adjusted = vector1 - mean1
    vector2_adjusted = vector2 - mean2

    # Compute the cosine similarity between the adjusted vectors
    dot_product = np.dot(vector1_adjusted, vector2_adjusted)
    norm1 = np.linalg.norm(vector1_adjusted)
    norm2 = np.linalg.norm(vector2_adjusted)

    # Avoid division by zero by checking if the norms are not zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def compute_mean_absolute_error(predictions, actual):
    """
    Calculate mean absolute error between predicted and actual values.

    Args:
        predictions (numpy.ndarray): Predicted values.
        actual (numpy.ndarray): Actual values.

    Returns:
        float: Mean absolute error.
    """
    absolute_errors = np.abs(predictions - actual)
    return np.mean(absolute_errors)

def k_nearest_neighbors_regression(X_train, X_test, y_train, k, similarity_func):
    """
    Perform k-nearest neighbors regression to predict values for test data.

    Args:
        X_train (list): Training data points.
        X_test (list): Test data points for prediction.
        y_train (numpy.ndarray): Labels for the training data.
        k (int): Number of neighbors to consider.
        similarity_func (function): Function to calculate similarity between data points.

    Returns:
        numpy.ndarray: Predicted values for the test data.
    """
    predictions = []

    for i in range(len(X_test)):
        # Calculate similarities for all training samples
        similarities = np.array([similarity_func(X_test[i], x) for x in X_train])
        # Find indices of the k-nearest neighbors
        nearest_indices = np.argpartition(-similarities, k)[:k]
        # Get the corresponding labels of the nearest neighbors
        nearest_labels = y_train[nearest_indices]
        # Calculate the predicted value as the mean of the nearest neighbor labels
        predicted_value = np.mean(nearest_labels)
        predictions.append(predicted_value)

    return np.array(predictions)

def weighted_k_nearest_neighbors_regression(X_train, X_test, y_train, k, similarity_func):
    """
    Perform weighted k-nearest neighbors regression to predict values for test data.

    Args:
        X_train (list): Training data points.
        X_test (list): Test data points for prediction.
        y_train (numpy.ndarray): Labels for the training data.
        k (int): Number of neighbors to consider.
        similarity_func (function): Function to calculate similarity between data points.

    Returns:
        numpy.ndarray: Predicted values for the test data.
    """
    predictions = []

    for i in range(len(X_test)):
        # Calculate similarities for all training samples
        similarities = np.array([similarity_func(X_test[i], x) for x in X_train])
        # Find indices of the k-nearest neighbors
        nearest_indices = np.argpartition(-similarities, k)[:k]
        # Get the corresponding labels and similarities of the nearest neighbors
        nearest_labels = y_train[nearest_indices]
        nearest_similarities = similarities[nearest_indices]
        # Calculate the weighted predicted value
        weighted_value = np.sum(nearest_labels * nearest_similarities) / (np.sum(nearest_similarities) if np.sum(nearest_similarities) > 0 else 1)
        predictions.append(weighted_value)

    return np.array(predictions)


# %%
scaled_animes.head()

# %%


# Now you can use the 'drop' method to extract X_train and y_train
X_train = scaled_animes.drop(['rating'], axis=1)  # features
y_train = scaled_animes['rating']  # target varia



# Now you can use the 'drop' method to extract X_train and y_train
X_test = test_scaled_animes.drop(['rating'], axis=1)  # features
y_test = test_scaled_animes['rating']  # target varia

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Set the number of K-fold splits and random seed
n_splits = 5
random_seed = 42

# Define the K-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# Define a function to plot Mean Absolute Error (MAE) for different K values
def plot_mae(results_df):
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g']  # Line colors for the two models
    markers = ['o', 's']  # Markers for the two models

    for i, (model, label) in enumerate(zip(['KNN', 'Weighted KNN'], ['K-Nearest Neighbors (KNN)', 'Weighted K-Nearest Neighbors (Weighted KNN)'])):
        plt.plot(results_df['K Value'], results_df[f'MAE ({model})'], 
                 color=colors[i], marker=markers[i], label=label)

    plt.title('K Value vs. Mean Absolute Error')
    plt.xlabel('K Value')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(results_df['K Value'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mae_plot.png', dpi=300)  # Save the plot as an image
    plt.show()

# Initialize a list to store the results as dictionaries
results = []

# Explore different values of K
k_values = [3, 5, 7]

# Loop over different values of K
for k in k_values:
    knn_mae, weighted_knn_mae = [], []

    # Perform K-fold cross-validation
    for train_idx, test_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

        # Get predictions from the models
        knn_preds = k_nearest_neighbors_regression(X_train_fold.to_numpy(), X_val_fold.to_numpy(), y_train_fold.to_numpy(), k, calculate_adjusted_cosine_similarity)
        weighted_knn_preds = weighted_k_nearest_neighbors_regression(X_train_fold.to_numpy(), X_val_fold.to_numpy(), y_train_fold.to_numpy(), k, calculate_adjusted_cosine_similarity)

        # Calculate the MAE for this fold
        knn_mae.append(mean_absolute_error(knn_preds, y_val_fold))
        weighted_knn_mae.append(mean_absolute_error(weighted_knn_preds, y_val_fold))

    # Record the average MAE for this value of K
    results.append({
        'K Value': k,
        f'MAE (KNN)': np.mean(knn_mae),
        f'MAE (Weighted KNN)': np.mean(weighted_knn_mae)
    })

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results)

# Call the plot function to visualize the results
plot_mae(results_df)

print(results_df)


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Set the number of K-fold splits and a random seed for reproducibility
n_splits = 5
random_seed = 42

# Define the K-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# Define a function to plot Mean Absolute Error (MAE) for different K values
def plot_mae(results_df):
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g']  # Line colors for the two models
    markers = ['o', 's']  # Markers for the two models

    for i, (model, label) in enumerate(zip(['KNN', 'Weighted KNN'], ['K-Nearest Neighbors (KNN)', 'Weighted K-Nearest Neighbors (Weighted KNN)'])):
        plt.plot(results_df['K Value'], results_df[f'MAE ({model})'], 
                 color=colors[i], marker=markers[i], label=label)

    plt.title('K Value vs. Mean Absolute Error')
    plt.xlabel('K Value')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(results_df['K Value'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mae_plot.png', dpi=300)  # Save the plot as an image
    plt.show()

# Initialize a list to store the results as dictionaries
results = []

# Explore different values of K
k_values = [3, 5, 7]

# Loop over different values of K
for k in k_values:
    knn_mae, weighted_knn_mae = [], []

    # Perform K-fold cross-validation
    for train_idx, test_idx in kf.split(X_test):
        X_test_fold, X_val_fold = X_test.iloc[train_idx], X_test.iloc[test_idx]
        y_test_fold, y_val_fold = y_test.iloc[train_idx], y_test.iloc[test_idx]

        # Get predictions from the models
        knn_preds = k_nearest_neighbors_regression(X_test_fold.to_numpy(), X_val_fold.to_numpy(), y_test_fold.to_numpy(), k, calculate_adjusted_cosine_similarity)
        weighted_knn_preds = weighted_k_nearest_neighbors_regression(X_test_fold.to_numpy(), X_val_fold.to_numpy(), y_test_fold.to_numpy(), k, calculate_adjusted_cosine_similarity)

        # Calculate the MAE for this fold
        knn_mae.append(mean_absolute_error(knn_preds, y_val_fold))
        weighted_knn_mae.append(mean_absolute_error(weighted_knn_preds, y_val_fold))

    # Record the average MAE for this value of K
    results.append({
        'K Value': k,
        f'MAE (KNN)': np.mean(knn_mae),
        f'MAE (Weighted KNN)': np.mean(weighted_knn_mae)
    })

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results)

# Call the plot function to visualize the results
plot_mae(results_df)

print(results_df)


# %%
"""
# ********
# REPORT 
# ********

"""

# %%
"""
> This report discusses the development and evaluation of an anime recommendation system using k nearest neighbors k nn and weighted k nn techniques with adjusted cosine similarity the dataset contains information about anime including their sources genres and types the goal of the system is to provide personalized anime recommendations to users
"""

# %%
"""
## Data Preprocessing
"""

# %%
"""
* Before implementing the recommendation system, we performed one-hot encoding on the categorical features: sources, genres, and types of animes. This encoding transforms categorical data into a binary format, making it suitable for k-NN algorithms.

* I removed two specific columns from the dataset: the "Name" of the anime and "users." These columns were not relevant to the collaborative filtering-based recommendation system we were building. By removing these columns, we ensured that only the essential data for our recommendation algorithm was retained.

* Following the one-hot encoding of categorical variables and the removal of irrelevant columns, we concatenated all the one-hot encoded categorical features with the remaining numerical variables. This created a unified dataset with both categorical and numerical features, which was used as input to the k-NN and weighted k-NN algorithms for anime recommendations.
"""

# %%
"""
## Methodology 

* Adjusted Cosine Similarity
  >We used adjusted cosine similarity as our similarity metric to measure the similarity between anime. This metric takes into account the user's rating behavior and helps mitigate rating biases.

* k-Nearest Neighbors (k-NN) and Weighted k-NN
  > We implemented both the k-NN and weighted k-NN algorithms to find the most similar animes to a given anime based on their adjusted cosine similarities. k represents the number of nearest neighbors considered.


"""

# %%
"""
## Result

"""

# %%
"""
# TRAIN DATA RESULT 

> ![Image ](Train.png)


# TEST DATA RESULT 

> ![Image ](Test.png)


* We evaluated the recommendation system on both the training and test datasets using different values of k (k = 3, 5, 7). The primary evaluation metric used was Mean Absolute Error (MAE), which quantifies the difference between predicted and actual user ratings.


"""

# %%
"""
## Discussion 

* Choice of k
 > The choice of the number of neighbors (k) has a significant impact on the performance of the recommendation system. We can observe that as k increases, the MAE decreases for both k-NN and weighted k-NN. This is expected, as a larger k allows for a broader set of neighbors to influence recommendations. However, the decrease in MAE is more prominent in the test dataset, suggesting that a smaller k may be better for generalization. The results show that k = 7 offers the lowest MAE for both k-NN and weighted k-NN on the test data.

* Weighted k-NN
 >The difference in MAE between k-NN and weighted k-NN is minimal, indicating that considering the weighted k-NN approach might not significantly improve the recommendation system's performance in this scenario. Weighted k-NN can be computationally more expensive, so the decision to use it should consider trade-offs in computational resources.
"""

# %%
"""
## Conclusion

In conclusion, we have successfully developed an anime recommendation system using k-NN and weighted k-NN with adjusted cosine similarity and one-hot encoding for categorical features. The choice of k significantly influences the system's performance, with k = 7 providing the lowest MAE on the test dataset. We also observed that weighted k-NN did not yield a substantial improvement in MAE compared to k-NN. Further enhancements can be explored, such as incorporating user-specific features and fine-tuning the choice of k for specific users.


"""

# %%
