import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ************************************************************** #
#                                                                #
#                        DATA WRANGLING                          #
#                                                                #
# ************************************************************** #

"""
OBS: My project is made with the "interactive window" for python instead of a notebook.
"""

# --------------------------------------------------------------
#                        Data Exploration
# --------------------------------------------------------------

# Setting index to be the column "id".
df = pd.read_csv("../data/listings.csv", index_col="id")

# Identifying missing values, which will be cleaned in the next section, neighbourhood_group, price, last_review, reviews_per_month, license.
df.info()

# Displaying basic statistical summaries of numerical columns, aswell as categorical columns:
df.describe()
df.mode().iloc[0]

# Initial Visualization plots:

"""
This Scatter Plot uses latitude and longitude for positioning,
while the color and size represent the price, giving a visual insight
into how prices are distributed geographically in Copenhagen.
"""
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="longitude",
    y="latitude",
    data=df,
    hue="price",
    size="price",
    alpha=0.6,
    sizes=(20, 200),
)
plt.title("Price Distribution by Location")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

"""
This box plot helps identify the central tendency of prices within each roome type,
along with the presence of outliers.
"""
plt.figure(figsize=(12, 6))
sns.boxplot(x="room_type", y="price", data=df)
plt.title("Price Distribution by Room Type")
plt.show()


# --------------------------------------------------------------
#                         Data Cleaning
# --------------------------------------------------------------

# I will now drop columns that are less than 50% count, and afterwards identify and remove outliers, "neighbourhood_group" and "license":
df_cleaned = df.dropna(axis=1, how="all")

# Handling missing values:
"""
First I've decided to remove the data that doesn't have a price,
instead of filling the values without "price", to avoid skewed data.

Secondly "reviews_per_month" will be filled with "0" reviews per month,
assuming no reviews mean the listing is new or not popular.

Lastly, since "last_review" is a datetime, I will keep it as NaT.
"""
df_cleaned = df_cleaned.dropna(subset=["price"])
df_cleaned["reviews_per_month"] = df_cleaned["reviews_per_month"].fillna(0)


# Identifying Outliers
"""
I will focus on "price, minimum_nights, number_of_reviews", 
since they have extreme outliers.
"""
plt.figure(figsize=(15, 5))
columns_to_plot = ["price", "minimum_nights", "number_of_reviews"]
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(1, len(columns_to_plot), i)
    sns.boxplot(y=df[column])
    plt.title(f"Box plot of {column}")

plt.tight_layout()
plt.show()

df_cleaned

df_cleaned.info()


# Removing Outliers using the IQR Range (function used in OLA-1):
def remove_outliers_iqr(dataset, col):
    """
    Function to mark values as outliers using the IQR (Interquartile Range) method.

    Explanation:
    A common method is to use the Interquartile Range (IQR),
    which is the range between the first quartile (25%) and the third quartile (75%)
    of the data. Outliers are often considered as data points that lie outside 1.5 times
    the IQR below the first quartile and above the third quartile.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtering the DataFrame to remove outliers:
    df_filtered = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    return df_filtered


# Removing outliers for each column:
columns_to_remove_outliers = ["price", "minimum_nights", "number_of_reviews"]
for column in columns_to_remove_outliers:
    df_cleaned = remove_outliers_iqr(df_cleaned, column)

# Checking the NaN Count again to assure the outliers has been removed.
df_cleaned.info()


# --------------------------------------------------------------
#                      Data Transformation
# --------------------------------------------------------------

df_transformed = df_cleaned.copy()

"""
Feature selection will focus on the numeric features: "price, minimum_nights,
number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365".

The categorical features will be "room_type, neighbourhood".
"""

# Feature Scaling / Standardization for numerical features:
from sklearn.preprocessing import StandardScaler

numeric_cols = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]
scaler = StandardScaler()
df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])


# Encoding categorical features using One-Hot Encoding:
df_transformed = pd.get_dummies(df_transformed, columns=["room_type", "neighbourhood"])
encoded_cols = list(df_transformed.columns.difference(df_cleaned.columns))


"""
I will now start clustering using K-Means.
The value of each feature within a centroid can give me insights 
into what characterizes each cluster's listings, 
such as higher prices, locations, types of rooms, etc.
"""
from sklearn.cluster import KMeans

# Combining numeric and encoded categorical columns:
all_features = numeric_cols + encoded_cols

# Elbow method to find optimal clusters:
wcss = []  # within-cluster-sum-of-square
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42
    )
    kmeans.fit(df_transformed[all_features])
    wcss.append(kmeans.inertia_)

# Plotting the elbow:
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Fitting the optimal amound of clusters:
kmeans = KMeans(n_clusters=6, random_state=42)
df_transformed["cluster"] = kmeans.fit_predict(df_transformed[all_features])


"""
To visualize the clusters formed by the KMeans algorithm I will use PCA,
to transform the data into a lower-dimensional space.
"""
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_transformed[all_features])

# Creating a new DataFrame with the principal components:
df_pca = pd.DataFrame(data=pca_features, columns=["pca_1", "pca_2"])

# Adding the cluster labels to this new DataFrame:
df_pca["Cluster"] = df_transformed["cluster"].values

# Getting the centroids from the model:
centroids = kmeans.cluster_centers_

# Transforming centropids using the same PCA to get them into the same space as the plot:
centroids_pca = pca.transform(centroids)

# Plotting the cluasters:
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="pca_1",
    y="pca_2",
    hue="Cluster",
    data=df_pca,
    palette=sns.color_palette("hsv", n_colors=len(df_pca["Cluster"].unique())),
    legend="full",
    alpha=0.7,
)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    s=100,
    c="black",
    marker="s",
    label="Centroid",
)

plt.title("KMeans Clusters visualized with PCA")
plt.show()


# --------------------------------------------------------------
#                      Data Modeling
# --------------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split

df_train = df_transformed.copy()

""" 
I have decided to exclude the columns "name, host_name, last_review" from the features before fitting,
since they are text objects and not relevant for this modeling process.

I find it appropriate to exclude these columns, since they don't contribute numerical value
or categorical information that would be useful for this prediction context.
"""
columns_to_exclude = ["name", "host_name", "last_review"]
df_train.drop(
    columns=[col for col in columns_to_exclude if col in df_train.columns], inplace=True
)

# Defining features:
X = df_train.drop(["price"], axis=1)
y = df_train["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.dtypes)


# Modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initializing the Random Forest Regressor:
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fitting the model on the training data:
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print(f"Root Mean Squared Error for Random Forest: {rmse_rf}")

# Feature Importance Analysis:
feature_importances = pd.DataFrame(
    rf_model.feature_importances_, index=X_train.columns, columns=["importance"]
).sort_values("importance", ascending=False)
print(feature_importances)


"""
The result is 0.81 in the context of predicting standardized prices for listings.
So on average it suggest that the model's predictions deviate from the actual standardized prices
by less than one std. I'm satisfied with this, since it indicates a relatively good performance.

In the end I've presented the feature importances as determined by the Random Forest model.
This analysis highlights the most significant features influencing the prediction of the prices.

If I had more time I would have validated and tuned the hyperparameters, and used techniques such as,
cross-validation. I could also have predicted the F1-score. 
Overall I'm satisfied with this small project's result.
"""
