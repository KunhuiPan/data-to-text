# Loading useful packages
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import json

# Loading data
shap_data = pd.read_csv("shap_data.csv", index_col=0)
shap_values = pd.read_csv("shap_values.csv", index_col=0)

ori_data = shap_data.copy()
shap_value = shap_values.copy()
# Check if there is duplicate columns from SHAP_data dataset and SHAP_values dataset and remove them
shap_value = shap_value.loc[:, ~shap_values.columns.duplicated()]
ori_data = ori_data.loc[:, ~shap_data.columns.duplicated()]
features = ori_data.columns


################# original data set preprocessing ########################
# Handel with columns of OneHotEncoding, revert the process of OneHotEncoding into labels
# in order to reduce the size of columns
class ConvertOHE:
    def __init__(self, df, prefix):
        self.df = df
        self.prefix = prefix

    def convert_ohe(self):
        for prefix in self.prefix:
            prefix_df = self.df.loc[:, self.df.columns.str.startswith(prefix)]
            self.df[prefix] = prefix_df.idxmax(axis=1).str.replace(prefix, '')
            self.df.drop(prefix_df.columns, axis=1, inplace=True)
            self.df[prefix] = self.df[prefix].str.slice(1)

        return self.df


prefixes = ['scenario', 'cust_risk_score', 'bcnf_type_Name', 'IntBusinessCd',
            'CustAddressCountryId', 'CustRiskCountryId', 'CustDomcCountryId',
            'CountryAmlRiskClassification', 'IsEeaCountry', 'IsEmergingMarketCountry',
            'IsEuApplicantCountry', 'IsEuCandidateCountry', 'IsEuCountry', 'IsOecdCountry',
            'IsTaxHavenCountry', 'Customer gender', 'Custtype']
OHE_convert = ConvertOHE(ori_data, prefixes)
ori_data = OHE_convert.convert_ohe()


# Handle with columns whose values are abbreviation, to make values understandable and meaningful
class AbbreviationProcessor:
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns

        self.country_map = {
            'de': 'Germany',
            'dk': 'Denmark',
            'fi': 'Finland',
            'fr': 'France',
            'gb': 'United Kingdom',
            'nl': 'Netherlands',
            'no': 'Northern Ireland',
            'se': 'Norway',
            'tr': 'Turkey',
            'us': 'United Kingdom',
            'infrequent_sklearn': 'infrequent country'
        }

        self.business_type_map = {
            'individ': 'individual',
            'organisa': 'organisation',
            'infrequent_sklearn': 'infrequent customer type'
        }

        self.gender_map = {
            'm': 'male',
            'f': 'female',
            'n': 'neutral gender',
            '_unknown': 'unknown gender'
        }

        self.country_risk_stratification = {
            'a': 'high risk country',
            'b': 'medium risk country',
            'c': 'low risk country',
            'infrequent_sklearn': 'uncommon country risk classification'
        }

        self.business_card_map = {
            'infrequent_sklearn': 'uncommon business card',
            'privat': 'private card',
        }

        self.risk_score_stratification = {
            'high': 'high risk score',
            'median': 'median risk score',
            'low': 'low risk score',
            'infrequent_sklearn': 'uncommon risk score'
        }

        # self.y_map = {
        #     1: 'false alert',
        #     0: 'real alert'
        # }

    def country_mapping(self):
        for col in self.columns:
            self.df[col] = self.df[col].replace(self.country_map)
        return self.df

    def business_mapping(self):
        for col in self.columns:
            self.df[col] = self.df[col].replace(self.business_type_map)
        return self.df

    def gender_mapping(self):
        for col in self.columns:
            self.df[col] = self.df[col].replace(self.gender_map)
            return self.df

    def country_risk_classification(self):
        for col in self.columns:
            self.df[col] = self.df[col].replace(self.country_risk_stratification)
            return self.df

    def business_card_mapping(self):
        for col in self.columns:
            self.df[col] = self.df[col].replace(self.business_card_map)
            return self.df

    def risk_score_classification(self):
        for col in self.columns:
            self.df[col] = self.df[col].replace(self.risk_score_stratification)
            return self.df

    # def y_mapping(self):
    #     for col in self.columns:
    #         self.df[col] = self.df[col].replace(self.y_map)
    #         return self.df


country_col = ['CustAddressCountryId', 'CustRiskCountryId', 'CustDomcCountryId']
country_convert = AbbreviationProcessor(ori_data, country_col)
ori_data = country_convert.country_mapping()

business_col = ['Custtype']
business_convert = AbbreviationProcessor(ori_data, business_col)
ori_data = business_convert.business_mapping()

gender_col = ['Customer gender']
country_convert = AbbreviationProcessor(ori_data, gender_col)
ori_data = country_convert.gender_mapping()

country_risk_col = ['CountryAmlRiskClassification']
country_convert = AbbreviationProcessor(ori_data, country_risk_col)
ori_data = country_convert.country_risk_classification()

business_card_col = ['IntBusinessCd']
country_convert = AbbreviationProcessor(ori_data, business_card_col)
ori_data = country_convert.business_card_mapping()

bcnf_type_col = ['bcnf_type_Name']
country_convert = AbbreviationProcessor(ori_data, bcnf_type_col)
ori_data = country_convert.business_card_mapping()

# y = ['y']
# country_convert = AbbreviationProcessor(ori_data, y)
# ori_data = country_convert.y_mapping()
cols = ori_data.columns.tolist()
cols.remove('y')
cols.append('y')
ori_data = ori_data[cols]


# ori_data.to_csv('df1.csv')

################# Shap data set preprocessing ########################
# To sum the shap value of OneHotEncoding so that the size of processed original data set
# equals to the size of the new SHAP data set
class ConvertSHAP:
    def __init__(self, df, prefix):
        self.df = df
        self.prefix = prefix

    def sum_shap_value(self):
        for prefix in self.prefix:
            prefix_df = self.df.loc[:, self.df.columns.str.startswith(prefix)]
            self.df[prefix] = prefix_df.sum(axis=1)
            self.df.drop(prefix_df.columns, axis=1, inplace=True)

        return self.df


convert_shap = ConvertSHAP(shap_value, prefixes)
shap_value_convert = convert_shap.sum_shap_value()
cols_shap = shap_value_convert.columns.tolist()
cols_shap.remove('prediction')
cols_shap.append('prediction')
shap_value_convert = shap_value_convert[cols_shap]
# shap_value_convert.to_csv('df2.csv')


################# Ground truth generation ########################
# Load datasets
file_path1 = "df1.csv"
df_input = pd.read_csv(file_path1)
file_path2 = "df2.csv"
df_shap = pd.read_csv(file_path2)

df_shap = df_shap.drop(df_shap.columns[0], axis=1)
# df_shap = df_shap.iloc[:10000]  # 10000

df_input = df_input.drop(df_input.columns[0], axis=1)


# df_input = df_input.iloc[:10000]  # 10000


# Normalizing each feature's importance except the prediction shap value
# to measure the contribution to the prediction for each feature
def normalized_contributions(shap_data):
    feature_shap = shap_data.loc[:, shap_data.columns != 'prediction']
    features = feature_shap.columns.to_numpy()

    shap_arr = feature_shap.to_numpy()
    sum_shap = np.sum(np.abs(shap_arr), axis=1)
    nor_contribution = shap_arr / sum_shap[:, None]  # normalized importance of each feature

    nor_contrib_df = pd.DataFrame(nor_contribution, columns=features)
    nor_contrib_df['prediction'] = shap_data.loc[:, 'prediction']  # put prediction value back

    return nor_contrib_df


nor_contrib_df = normalized_contributions(df_shap)
nor_contrib_df.to_csv('nor_contrib_df.csv')

for index, row in nor_contrib_df.iterrows():
    abs_values = row.abs()  # Calculate the absolute value of each row
    max_abs_values = abs_values.nlargest(11)  # Obtain the 11th elements with the largest absolute value
    row[~abs_values.isin(max_abs_values)] = 0  # Set the other cells to 0

output_file_path = "data_shap.csv"
nor_contrib_df.to_csv(output_file_path, index=False)

zero_cells = np.where(nor_contrib_df == 0)

for row, col in zip(zero_cells[0], zero_cells[1]):
    df_input.iat[row, col] = np.nan

output_file_path = "data_prepro.csv"
df_input.to_csv(output_file_path, index=False)

# Create an empty list to store column names with non-null values
non_empty_columns = []

# Iterate through each column of the dataframe
for column in df_input.columns:
    # Check each column for non-null values
    if df_input[column].notnull().any():
        # If there is a non-null value, add the column name to the non-empty_columns list
        non_empty_columns.append(column)

# Output a list of non_empty_columns with column names that have non-null values
print(non_empty_columns)

df_shap = pd.read_csv('data_shap.csv')
df_input = pd.read_csv('data_prepro.csv')

# Modify the column name of y to prediction
df_input.rename(columns={'y': 'prediction'}, inplace=True)
# template-based method to generate text summary
def generate_output(row):
    output_list = []
    for col, value in row.items():
        if not pd.isnull(value):
            shap_value = df_shap.loc[row.name, col]
            if col == " total amount transfered":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == " total amount in transactions":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == " total amount received":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "max  of amount transfered":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "average of  of total amount":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "average  of total amount received":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "average  of total amount transfered":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "average  of total amount transfered":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "Number of transactions in case":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} transaction with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} transaction with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} transaction with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} transaction with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} transaction with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} transaction with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "Number of incoming transactions in case":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} incoming transaction with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} incoming transaction with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} incoming transaction with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} incoming transaction with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} incoming transaction with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} incoming transaction with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "Number of outgoing transactions in case":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} outgoing transaction with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} outgoing transaction with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} outgoing transaction with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} outgoing transaction with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"in {value:.0f} outgoing transaction with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"in {value:.0f} outgoing transaction with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "Std of total amount received":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The standard deviation of total received amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The standard deviation of total received amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The standard deviation of total received amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The standard deviation of total received amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The standard deviation of total received amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The standard deviation of total received amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "Std of total amount transfered":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The standard deviation of total transfered amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The standard deviation of total transfered amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The standard deviation of total transfered amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The standard deviation of total transfered amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The standard deviation of total transfered amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The standard deviation of total transfered amount is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "max  of amount received":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "max  of amount transfered":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"{col} is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "cust_risk_score":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer has a risk score classified as {value} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer has a risk score classified as {value} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer has a risk score classified as {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer has a risk score classified as {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer has a risk score classified as {value} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer has a risk score classified as {value} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "bcnf_type_Name":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "IntBusinessCd":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The bussiness card is {value} category with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The bussiness card is {value} category with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The bussiness card is {value} category with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The bussiness card is {value} category with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The bussiness card is {value} category with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The bussiness card is {value} category with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "NumberLinkedCustomers":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"Moreover, there is only {value:.0f} linked customer with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"Moreover, there is only {value:.0f} linked customer with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"Moreover, there is only {value:.0f} linked customer with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"Moreover, there is only {value:.0f} linked customer with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"Moreover, there is only {value:.0f} linked customer with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"Moreover, there is only {value:.0f} linked customer with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")


            elif col == "CustAddressCountryId":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "CustRiskCountryId":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "scenario":  # 检查是否是"scenario"列
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"This case is under scenario {value} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"This case is under scenario {value} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"This case is under scenario {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"This case is under scenario {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"This case is under scenario {value} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"This case is under scenario {value} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "YearsInDB":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer has been in the database for {value} years with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer has been in the database for {value} years with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer has been in the database for {value} years with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer has been in the database for {value} years with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer has been in the database for {value} years with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer has been in the database for {value} years with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "Customer age":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer's age is {value:.0f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer's age is {value:.0f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer's age is {value:.0f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer's age is {value:.0f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer's age is {value:.0f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer's age is {value:.0f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "CASHDEPOCASHDEPO2":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer has made a cash deposit with an amount of {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer has made a cash deposit with an amount of {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer has made a cash deposit with an amount of {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer has made a cash deposit with an amount of {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer has made a cash deposit with an amount of {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer has made a cash deposit with an amount of {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "TRANSACTCRBINCOM1":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(01) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(01) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(01) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(01) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(01) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(01) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "TRANSACTCRBINCOMY2":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(02) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(02) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(02) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(02) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(02) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the incoming cross-border transfers amount(02) to {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "TRANSACTCRBOUTGO1":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(01) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(01) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(01) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(01) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(01) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(01) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "TRANSACTCRBOUTGO2":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(02) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(02) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(02) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(02) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(02) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"Additionally, the outgoing cross-border transfers amount(02) is {value:.2f} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "bcnf_type_Name":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"and their business card type is considered as {value} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "CustAddressCountryId":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their address country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "CustRiskCountryId":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their risk country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "CustDomcCountryId":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their domestic country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their domestic country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their domestic country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their domestic country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their domestic country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} in their domestic country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "CountryAmlRiskClassification":
                if abs(shap_value) > 0.1 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                elif abs(shap_value) > 0.1 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                elif abs(shap_value) > 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                elif abs(shap_value) > 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                elif abs(shap_value) < 0.02 and shap_value > 0:
                    output_list.append(
                        f"The customer is associated with the {value} with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                elif abs(shap_value) < 0.02 and shap_value < 0:
                    output_list.append(
                        f"The customer is associated with the {value} with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "IsEmergingMarketCountry":
                if value == "FALSE":
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is not from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is not from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
                else:
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is from emerging market country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "IsEuCandidateCountry":
                if value == "FALSE":
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is not EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is not EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
                else:
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is EU candidate country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "IsEuCountry":
                if value == "FALSE":
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is not EU country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is not EU country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not EU country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not EU country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not EU country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not EU country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
                else:
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is EU country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is EU country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is EU country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is EU country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is EU country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is EU country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "IsOecdCountry":
                if value == "FALSE":
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is not OECD country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is not OECD country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not OECD country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not OECD country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not OECD country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not OECD country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
                else:
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is OECD country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is OECD country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is OECD country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is OECD country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is OECD country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is OECD country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "IsTaxHavenCountry":
                if value == "FALSE":
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is not tax haven countryy with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is not tax haven country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not tax haven country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not tax haven country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not tax haven country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not tax haven country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
                else:
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is tax haven country with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is tax haven country with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is tax haven country with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is tax haven country with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is tax haven country with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is tax haven country with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
            elif col == "IsEeaCountry":
                if value == "FALSE":
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is not from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is not from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is not from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is not from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
                else:
                    if abs(shap_value) > 0.1 and shap_value > 0:
                        output_list.append(
                            f"The customer is from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a significant positive contribution.")
                    elif abs(shap_value) > 0.1 and shap_value < 0:
                        output_list.append(
                            f"The customer is from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a significant negative contribution.")
                    elif abs(shap_value) > 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a moderate positive impact.")
                    elif abs(shap_value) > 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a moderate negative impact.")
                    elif abs(shap_value) < 0.02 and shap_value > 0:
                        output_list.append(
                            f"The customer is from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a slight positive impact.")
                    elif abs(shap_value) < 0.02 and shap_value < 0:
                        output_list.append(
                            f"The customer is from the European Economic Area(EEA) with the effect of {shap_value * 100:.2f}%, indicating a slight negative impact.")
                        # output_list.append(f"{col} is {value}, corresponding shap value is {shap_value}")
            else:
                if value == 1:
                    if abs(shap_value) >= 0.85:
                        output_list.append(
                            f"The probability of this case being false alert is assessed as high risk.")
                    elif 0.5 <= abs(shap_value) < 0.8:
                        output_list.append(
                            f"The probability of this case being false alert is assessed as medium risk.")
                    else:
                        output_list.append(
                            f"The probability of this case being false alert is assessed as a low risk.")
                else:
                    if abs(shap_value) >= 0.85:
                        output_list.append(
                            f"The probability of this case being fraudulent is assessed as a high risk.")
                    elif 0.5 <= abs(shap_value) < 0.8:
                        output_list.append(
                            f"The probability of this case being fraudulent is assessed as a medium risk.")
                    else:
                        output_list.append(
                            f"The probability of this case being fraudulent is assessed as a low risk.")

    return " ".join(output_list)


# generate the output paragraph for each row and add it to a new column 'output' in df_input
df_input['summary'] = df_input.apply(generate_output, axis=1)

# 5. Save the updated dataframe with the output paragraph to a new CSV file
df_input.to_csv('data_prepro_with_output.csv', index=False)

df_shap = pd.read_csv('data_shap.csv')
df_input = pd.read_csv("data_prepro_with_output.csv")

# Split the data into train, validation, and test sets
# train_data = df_input.iloc[:8853]
# val_data = df_input.iloc[8854:11068]
# test_data = df_input.iloc[11069:13283]


# train_data = df_input.iloc[:73778]
# val_data = df_input.iloc[73779:92224]
# test_data =  df_input.iloc[92225:110667]

train_data = df_input.iloc[:50000]
test_data_reference = df_input.iloc[50000:51000]
test_data = df_input.iloc[50000:51000]

# Function to convert train and validation data to JSON format with both 'q' and 'a'
def csvTojson(datafile, output_file):
    # Iterate over each row in the DataFrame
    data = []
    for index, row in datafile.iterrows():
        q = f"Please provide a description of the data in the following financial domain\n"
        # Iterate over each column and its value in the current row
        for col, value in row.iteritems():
            if pd.notnull(value) and col not in ["summary"]:
                shap_value = df_shap.loc[index, col]
                # Exclude the 'summary' column and construct the 'q' string
                if col == "Number of transactions in case":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "NumberLinkedCustomers":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "Number of incoming transactions in case":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "Number of outgoing transactions in case":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "NumberLinkedCustomers":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "YearsInDB":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "Customer age":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "prediction":
                    q += f"{col}: {value:.0f} probability: {shap_value * 100:.2f}%"
                elif col in (' total amount in transactions', ' total amount received', ' total amount transfered',
                             'average  of total amount received', 'average  of total amount transfered',
                             'average of  of total amount',
                             'Std of total amount received', 'Std of total amount transfered',
                             'max  of amount received',
                             'max  of amount transfered', 'CASHDEPOCASHDEPO2', 'TRANSACTCRBOUTGO1', 'TRANSACTCRBOUTGO2',
                             'TRANSACTCRBINCOM1', 'TRANSACTCRBINCOMY2'):
                    q += f"{col}: {value:.2f} effect: {shap_value * 100:.2f}%"
                else:
                    q += f"{col}: {value} effect: {shap_value * 100:.2f}%\n"

        # Get the value from the 'summary' column as the answer 'a'
        a = row['summary']
        # Append the dictionary with 'q' and 'a' to the 'data' list
        data.append({"q": q, "a": a})
    # Write each dictionary as a separate JSON object to a JSON file
    with open(output_file, 'w') as f:
        for i in data:
            json.dump(i, f)
            f.write('\n')


# Convert train and validation data to JSON with 'q' and 'a'
# csvTojson(train_data, 'train_QA.json')
csvTojson(test_data_reference, 'test_data_reference.json')


# Function to convert test data to JSON format with only 'q' (without 'a')
def csvTojsonTest(datafile, output_file):
    # Iterate over each row in the DataFrame
    data = []
    for index, row in datafile.iterrows():
        q = f"Please provide a description of the data in the following financial domain\n"
        # Iterate over each column and its value in the current row
        for col, value in row.items():
            # Exclude the 'summary' column and construct the 'q' string
            if pd.notnull(value) and col not in ["summary"]:
                shap_value = df_shap.loc[index, col]
                # Exclude the 'summary' column and construct the 'q' string
                if col == "Number of transactions in case":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "NumberLinkedCustomers":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "Number of incoming transactions in case":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "Number of outgoing transactions in case":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "NumberLinkedCustomers":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "YearsInDB":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "Customer age":
                    q += f"{col}: {value:.0f} effect: {shap_value * 100:.2f}%\n"
                elif col == "prediction":
                    q += f"{col}: {value:.0f} probability: {shap_value * 100:.2f}%"
                elif col in (' total amount in transactions', ' total amount received', ' total amount transfered',
                             'average  of total amount received', 'average  of total amount transfered',
                             'average of  of total amount',
                             'Std of total amount received', 'Std of total amount transfered',
                             'max  of amount received',
                             'max  of amount transfered', 'CASHDEPOCASHDEPO2', 'TRANSACTCRBOUTGO1', 'TRANSACTCRBOUTGO2',
                             'TRANSACTCRBINCOM1', 'TRANSACTCRBINCOMY2'):
                    q += f"{col}: {value:.2f} effect: {shap_value * 100:.2f}%"
                else:
                    q += f"{col}: {value} effect: {shap_value * 100:.2f}%\n"
        # Append the dictionary with only 'q' to the 'data' list
        data.append({"q": q})

    # Write each dictionary as a separate JSON object to a JSON file
    with open(output_file, 'w') as f:
        for i in data:
            json.dump(i, f)
            f.write('\n')


# Convert test data to JSON with only 'q' (without 'a')
# csvTojsonTest(test_data, 'test.json')




