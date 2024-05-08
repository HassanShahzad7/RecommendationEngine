# Importing Libraries
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, ndcg_score

# Load and preprocess data
data = pd.read_csv('data.csv', encoding='ISO-8859-1')


def preprocessing_data(df):
    """
        Preprocesses e-commerce transactional data by converting invoice dates to
        datetime, removing rows with missing essential fields, and adding new features.

        Args:
            df (pandas.DataFrame): Input dataframe containing transactional data.

        Returns:
            pandas.DataFrame: Preprocessed dataframe with additional date and stock code features.
        """
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.dropna(subset=['Description', 'CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDay'] = df['InvoiceDate'].dt.dayofweek
    df['InvoiceMonth'] = df['InvoiceDate'].dt.month
    df['InvoiceHour'] = df['InvoiceDate'].dt.hour
    df['StockCodeCat'] = df['StockCode'].astype('category').cat.codes
    return df


def sparse_matrix(df):
    """
        Constructs a sparse item-user matrix based on transaction data and computes item-item
        similarity using cosine similarity.

        Args:
            df (pandas.DataFrame): Input dataframe containing transactional data.

        Returns:
            tuple: A tuple containing:
                - item_user_matrix (scipy.sparse.csr_matrix): Sparse matrix with items and users.
                - item_similarity (scipy.sparse.csr_matrix): Sparse matrix with item-item similarities.
        """
    df['StockCodeCat'] = pd.Categorical(df['StockCode']).codes
    item_user_matrix = csr_matrix((df['Quantity'], (df['StockCodeCat'], df['CustomerID'])))

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(item_user_matrix.T)

    # Convert cosine similarity to sparse matrix
    item_similarity = csr_matrix(cosine_sim)

    return item_user_matrix, item_similarity


def get_item_recommendations(df, items_bought, top_n=10):
    """
        Generates item recommendations for a user based on items they've previously purchased.
        It performs Collaborative Filtering
        Args:
            df (pandas.DataFrame): Dataframe containing transactional data.
            items_bought (list): List of stock codes representing items the user has already bought.
            top_n (int): Number of top recommendations to return. Defaults to 10.

        Returns:
            list: A list of recommended stock codes based on item similarity or top-selling items.
        """
    _, item_similarity = sparse_matrix(df)
    # Convert stock codes to their categorical indices
    stock_code_map = {code: idx for idx, code in enumerate(df['StockCode'].astype('category').cat.categories)}
    items_indices = [stock_code_map.get(code) for code in items_bought if stock_code_map.get(code) is not None]

    if items_indices:
        # Extract the relevant rows from the similarity matrix
        similarity_matrix = item_similarity[items_indices, :]

        # Summing and converting the similarity scores appropriately
        if scipy.sparse.issparse(similarity_matrix):
            similarity_scores = similarity_matrix.sum(axis=0).A1  #.A1 returns a flattened numpy array
        elif isinstance(similarity_matrix, np.matrix):
            similarity_scores = similarity_matrix.sum(axis=0).A1  #.A1 returns a flattened numpy array
        else:
            similarity_scores = np.array(similarity_matrix.sum(axis=0)).ravel()

        # Identify the top items by their indices
        top_items_indices = similarity_scores.argsort()[-top_n:][::-1]

        # Ensure the indices are within bounds
        top_items_indices = [i for i in top_items_indices if i < len(df['StockCode'].astype('category').cat.categories)]

        # Remove items that the user has already bought
        top_items_indices = [i for i in top_items_indices if df['StockCode'].astype('category').cat.categories[i] not in items_bought]

        # If there are no new items, return the top selling items
        if not top_items_indices:
            return recommend_top_items(df, top_n)

        # Fetch the stock codes for the top indices
        top_stock_codes = [df['StockCode'].astype('category').cat.categories[i] for i in top_items_indices]
        return top_stock_codes
    else:
        return recommend_top_items(df, top_n)




def recommend_top_items(df, n=10):
    """
        Recommends the top-selling items based on overall popularity.

        Args:
            df (pandas.DataFrame): Dataframe containing transactional data.
            n (int): Number of top items to return. Defaults to 10.

        Returns:
            list: A list of the most popular stock codes.
        """
    item_popularity = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    return item_popularity.head(n).index.tolist()


def recommend_products(df, user_id):
    """
        Provides hybrid product recommendations for a specific user.
        If the User is new then in that case uses Cold Start Strategy and recommends top most selling items to the customer.
        If the User is existing and has made transactions then recommends items based on his purchases.

        Args:
            df (pandas.DataFrame): Dataframe containing transactional data.
            user_id (int): The ID of the user for whom recommendations are generated.

        Returns:
            list: A list of stock codes representing the recommended products for the user.
        """
    user_items = df[df['CustomerID'] == user_id]['StockCode'].unique()
    if user_id not in df['CustomerID'].unique():
        return recommend_top_items(df, 10)
    else:
        return get_item_recommendations(df, user_items, 10)


def split_data(df):
    """
    Split data into training and testing sets based on a customer grouping.
    """
    # Split customers into training and test
    unique_customers = df['CustomerID'].unique()
    train_customers, test_customers = train_test_split(unique_customers, test_size=0.2, random_state=42)

    # Create training and test dataframes
    train_df = df[df['CustomerID'].isin(train_customers)]
    test_df = df[df['CustomerID'].isin(test_customers)]

    return train_df, test_df


def evaluate_recommendation(df_train, df_test, recommendation_fn, top_n=10):
    """
    Evaluate recommendation engine performance using precision and recall metrics.

    Args:
        df_train (pandas.DataFrame): Training dataframe.
        df_test (pandas.DataFrame): Testing dataframe.
        recommendation_fn (callable): Function that takes the training dataframe and user ID
                                      as input and returns recommended product IDs.
        top_n (int): Number of top recommendations.

    Returns:
        tuple: Average precision and recall scores.
    """
    precisions = []
    recalls = []

    for user_id in df_test['CustomerID'].unique():
        actual_items = set(df_test[df_test['CustomerID'] == user_id]['StockCode'])
        if not actual_items:
            continue

        recommended_items = set(recommendation_fn(df_train, user_id)[:top_n])

        if not recommended_items:
            precisions.append(0)
            recalls.append(0)
            continue

        # Calculate precision and recall
        true_positives = len(actual_items & recommended_items)
        precision = true_positives / len(recommended_items)
        recall = true_positives / len(actual_items)

        precisions.append(precision)
        recalls.append(recall)

    # Average metrics across all users
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    return avg_precision, avg_recall


# Example usage
# train_data, test_data = split_data(data)
# precision, recall = evaluate_recommendation(train_data, test_data, recommend_products, top_n=10)
# print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

# data = preprocessing_data(data)
# print(data.info())
# # Example call to recommend based on customer 17850's previous purchases
# print(recommend_products(data, 13047))

# a = ['85123A', '71053', '84406B', '84029G', '84029E', '22752', '21730', '22633', '22632']
# ['84077', '22197', '85099B', '84879', '85123A', '21212', '23084', '22492', '22616', '21977']
# ['22671', '22665', '22666', '22667', '22668', '22669', '22670', '22672', '22663', '22673']