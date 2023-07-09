# Loading useful packages
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

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
# shap_value_convert.to_csv('df2.csv')
