# useful packages
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Loading data
shap_data = pd.read_csv("shap_data.csv", index_col=0)
shap_values = pd.read_csv("shap_values.csv", index_col=0)

# Convert the dataframe into array
# SHAP values
obs_shap_values = shap_values.iloc[:, :147].to_numpy()
predictions = shap_values.iloc[:, -1].to_numpy()
feature_names = shap_values.iloc[:, :147].columns.to_numpy()
# SHAP data(original data)
feature_values = shap_data.iloc[:, :147].to_numpy()
y = shap_data.iloc[:, -1].to_numpy()


def case_likelihood_distribution(shap_data, shap_values):
    y_label = shap_data['y']
    shap_values_y0 = shap_values[y_label == 0]['prediction'].values.flatten()
    shap_values_y1 = shap_values[y_label == 1]['prediction'].values.flatten()

    # Plot the distribution of SHAP values for y_label = 0
    plt.figure(figsize=(10, 5))
    plt.hist(shap_values_y0, bins=30, alpha=0.5, color='blue', label='y=0')
    plt.xlabel('SHAP Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of SHAP Values for y=0')
    plt.legend()
    plt.show()

    # Plot the distribution of SHAP values for y_label = 1 (false positive)
    plt.figure(figsize=(10, 5))
    plt.hist(shap_values_y1, bins=30, alpha=0.5, color='blue', label='y=1')
    plt.xlabel('SHAP Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of SHAP Values for y=1')
    plt.legend()
    plt.show()

    return shap_values_y0, shap_values_y1


shap_values_y0, shap_values_y1 = case_likelihood_distribution(shap_data, shap_values)


# **Summary**: For cases which are false positive(y = 1), i.e. actual: no criminal activity, predicted: criminal
# case. The SHAP value varies from zero to one, Most cases have SHAP values greater than 0.5, but there are still a
# significant number of cases with values less than 0.5. This may indicate that we cannot determine whether a case is
# a crime by its SHAP value alone.

# **Important**: while SHAP shows the contribution or the importance of each feature on the prediction of the model,
# it does not evaluate the quality of the prediction itself.


# Calculate the contribution of each feature which is determined by the absolute SHAP-value of each feature
# The larger the value, the more contribution to the prediction.
def calculate_contributions(obs_shap_values, feature_names):
    """
    Relative contribution of each feature for every specific observation
    :param obs_shap_values: A numpy array of SHAP values with shape (n_samples, n_features).
    :param feature_names: The feature names.
    :return: A data frame contains relative importance of each feature for each observation.
    """
    # Calculate the total absolute SHAP values for each observation
    total_shap_values = np.sum(np.abs(obs_shap_values), axis=1)

    # Calculate feature contributions for all observations
    # The None keyword is a syntax in NumPy that increases an extra dimension so that we have 2D array.
    feature_contributions = obs_shap_values / total_shap_values[:, None]

    # Create a DataFrame with feature names as columns and observations as rows
    contributions_df = pd.DataFrame(feature_contributions, columns=feature_names)

    return contributions_df


# Get contribution data frame of each feature for every observation(row)
df_contributions = calculate_contributions(obs_shap_values, feature_names)


def get_top_features(contributions_df, num_features=15):
    """
    Get a sorted top features dictionary based on feature's absolute contribution value in descending order
    :param contributions_df: The contribution DataFrame
    :param num_features: The number of top contributing features, can be adjusted
    :return: A dictionary contains features and their values
    """
    # Get the absolute values of the contributions, the sum of all feature's contributions is 1
    absolute_contributions = np.abs(contributions_df.values)

    # Get the indices of the top features for each observation
    top_feature_indices = np.argsort(absolute_contributions, axis=1)[:, ::-1][:, :num_features]

    # Get the corresponding feature names
    feature_names = contributions_df.columns
    top_feature_names = np.array([feature_names[indices] for indices in top_feature_indices])

    # Get the corresponding feature values
    top_feature_values = np.array(
        [contributions_df.values[i, indices] for i, indices in enumerate(top_feature_indices)])

    # Create a dictionary to store the top features and their values for each observation
    top_features_dict = {}
    for i, names in enumerate(top_feature_names):
        values = top_feature_values[i]
        top_features_dict[i] = dict(zip(names, values))

    return top_features_dict


# Get the dictionary of top 15 features
top_features_dict = get_top_features(df_contributions)


def generate_explanation(top_features_dict, prediction, num_features=3):
    """
    Generate/show a text explainer based on top features dictionary with a basic rule-based approach,
    and contains top 3 features in the explanation for each observation
    :param top_features_dict: The feature dictionary
    :param num_features: The number of features that you want show in text explanation
    :return: Text explanations
    """
    explanations = []

    for obs_index, features_dict in top_features_dict.items():
        # Slice top n elements from top feature dictionary
        top_features = list(features_dict.keys())[:num_features]
        top_values = list(features_dict.values())[:num_features]

        explanation = f"For observation {obs_index + 1} with false alert probability {prediction[obs_index] * 100:.2f}%, the top {num_features} contributing features are: "
        for feature, value in zip(top_features, top_values):
            # If the SHAP value is positive which leads to positive contribution
            if value >= 0:
                explanation += f"{feature} increased the case risk score by {value * 100:.2f}%. "
            # If the SHAP value is negative which leads to negative contribution
            else:
                explanation += f"{feature} decreased the case risk score by {abs(value * 100):.2f}%. "
        explanations.append(explanation.strip())

    return explanations


# Show instances in text format
explanations = generate_explanation(top_features_dict, predictions, num_features=3)
print(explanations[0])
print('--------------------')
print(explanations[1])
print('--------------------')
print(explanations[2])
