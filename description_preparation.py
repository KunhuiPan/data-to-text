# useful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import shap

# Loading data
df1 = pd.read_csv("df1.csv", index_col=0)  # data set with original values
df2 = pd.read_csv("df2.csv", index_col=0)  # data set with shap values


# Normalizing each feature's importance except prediction shap value
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

nor_contrib_df = normalized_contributions(df2)


def false_alert_level(nor_contrib_df):

    def prediction(pre):
        if pre >= 0.85:
            return 'high likelihood'
        elif pre < 0.5:
            return 'median likelihood'
        else:
            return 'low likelihood'

    # Apply function to each row of the column
    nor_contrib_df['prediction'] = nor_contrib_df['prediction'].apply(prediction)

    return nor_contrib_df


nor_contrib_df = false_alert_level(nor_contrib_df)

################# Local importance explore #################
y = df1.loc[:, df1.columns == 'y']
ori_df = df1.loc[:, df1.columns != 'y']
prediction = df2.loc[:, df2.columns == 'prediction']
shap_df = df2.loc[:, df2.columns != 'prediction']
# An overview of first five rows
first_five_rows = shap_df.iloc[:5]
fig, axs = plt.subplots(5, 1, figsize=(10, 30))
min_value = np.min(first_five_rows.min())
max_value = np.max(first_five_rows.max())

for i in range(len(first_five_rows)):
    # Sort shap value
    sorted_value = first_five_rows.iloc[i].sort_values()

    # Create the bar plot
    sorted_value.plot(kind='barh', ax=axs[i])
    axs[i].set_xlim(min_value, max_value)
    axs[i].set_xlabel('SHAP Value')
    axs[i].set_title(f'Feature Importance for the Observation {i + 1} ')

plt.tight_layout()
plt.show()
# **Summary**: the length of each bar represents the magnitude of the feature's SHAP value.
# Each observation does not hold same important features. The magnitude varies case by case.
# i.e. for each specific observation, the important features are different.


################# Global importance explore #################
# Mean absolute SHAP value of each feature across the data.
mean_shap_value = shap_df.abs().mean().sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=mean_shap_value.values, y=mean_shap_value.index)
plt.title('Main Effects (mean absolute SHAP values)')
plt.show()


# **Summary**: the main features are transaction related(income, outgoing or both)
# and customer portrait related(individual, organization, residence country and country risk level, years in the bank, etc.).

# Note:The global effects of the features are ordered from the highest to the lowest effect
# on the prediction. By taking account the absolute SHAP value, it does not matter if a feature impact
# the prediction in a positive or negative way.


################# Description generation based on template and rule #################

class Description:
    def __init__(self, ori_data, shap_data):
        self.ori = ori_data
        self.shap = shap_data
        # self.features = shap_data.columns.tolist()

    # This template is to generate description which is related to observation's transaction
    def transact_risk_country_temp(self, index: int) -> str:
        ori_row = self.ori.iloc[index]
        shap_row = self.shap.iloc[index]
        features = self.shap.columns

        des = ''

        transact = ori_row[' total amount in transactions']
        income = ori_row[' total amount received']
        outgo = ori_row[' total amount transfered']
        countr_risk_level = ori_row.loc['CountryAmlRiskClassification']
        ave_transfer_amount = ori_row.loc['average  of total amount transfered']

        n_transact = ori_row['Number of transactions in case']
        n_income = ori_row['Number of incoming transactions in case']
        n_outgo = ori_row['Number of outgoing transactions in case']

        transact_shap = shap_row.loc[' total amount in transactions']
        income_shap = shap_row.loc[' total amount received']
        outgo_shap = shap_row.loc[' total amount transfered']
        # n_transact_shap = shap_row.loc[' total amount in transactions']
        n_income_shap = shap_row.loc[' total amount received']
        n_outgo_shap = shap_row.loc[' total amount transfered']
        risk_level_shap = shap_row.loc['CountryAmlRiskClassification']
        ave_transfer_amount_shap = shap_row.loc['average  of total amount transfered']

        sum_income_shap = income_shap + risk_level_shap
        sum_outgo_shap = outgo_shap + risk_level_shap
        sum_ave_outgo_shap = ave_transfer_amount_shap + n_outgo_shap
        sum_transact_shap = transact_shap + risk_level_shap

        impact1 = 'increased' if sum_income_shap > 0 else 'decreased' if sum_income_shap < 0 else 'remained constant'
        impact2 = 'raised' if outgo_shap > 0 else 'reduced' if outgo_shap < 0 else 'remained unchanged'

        impact3 = 'increased' if sum_outgo_shap > 0 else 'decreased' if sum_outgo_shap < 0 else 'remained constant'
        impact4 = 'raised' if sum_ave_outgo_shap > 0 else 'reduced' if sum_ave_outgo_shap < 0 else 'remained unchanged'

        impact5 = 'increased' if sum_transact_shap > 0 else 'decreased' if sum_transact_shap < 0 else 'remained constant'

        if n_transact != 0:  # check valid transaction
            if income != 0 and outgo == 0:  # only incoming transaction(received amount)
                sum_abs_income_shap = abs(income_shap) + abs(risk_level_shap)
                des += f'The total received amount of {round(income, 2) if income <= 1000000 else round(income / 1000000, 2)} {"DKK" if income <= 1000000 else "million DKK"} ' \
                       f'from {countr_risk_level} {impact1} the case risk score by {sum_abs_income_shap * 100:.2f}%. \n' \
                       f'In addition, the lack of outgoing transactions in the case {impact2} the escalation probability by {abs(outgo_shap) * 100:.2f}%.'

            if income == 0 and outgo != 0:  # only transfer transaction
                sum_abs_outgo_shap = abs(outgo_shap) + abs(risk_level_shap)
                sum_abs_ave_outgo_shap = abs(ave_transfer_amount_shap) + abs(n_outgo_shap)
                des += f'The total transfer amount of {round(outgo, 2) if outgo <= 1000000 else round(outgo / 1000000, 2)} {"DKK" if outgo <= 1000000 else "million DKK"} from {countr_risk_level} {impact3} the case risk score by {sum_abs_outgo_shap * 100:.2f}%. \n' \
                       f'In addition, the average transfer amount is {round(ave_transfer_amount, 2) if ave_transfer_amount <= 1000000 else round(ave_transfer_amount / 1000000, 2)} {"DKK" if ave_transfer_amount <= 1000000 else "million DKK"} in {int(n_outgo)} {"completion" if n_outgo <= 1 else "completions"} ' \
                       f'which {impact4} the risk probability by {sum_abs_ave_outgo_shap * 100:.2f}%.'

            if income != 0 and outgo != 0:  # both incoming and outgoing transaction
                sum_shap = transact_shap + risk_level_shap + income_shap + outgo_shap + n_income_shap + n_outgo_shap
                impact = 'increased' if sum_shap > 0 else 'decreased' if sum_shap < 0 else 'remained constant'
                sum_abs_shap = abs(transact_shap) + abs(risk_level_shap) + abs(income_shap) + abs(outgo_shap) + abs(
                    n_income_shap) + abs(n_outgo_shap)
                des += f'The total transaction amount from {countr_risk_level} is {round(transact, 2) if transact <= 1000000 else round(transact / 1000000, 2)} {"DKK" if transact <= 1000000 else "million DKK"}, ' \
                       f'with {round(income, 2) if income <= 1000000 else round(income / 1000000, 2)} {"DKK" if income <= 1000000 else "million DKK"} received in {int(n_income)} ' \
                       f'{"completion" if n_income <= 1 else "completions"},\nand {round(outgo, 2) if outgo <= 1000000 else round(outgo / 1000000, 2)} transferred in {int(n_outgo)} ' \
                       f'{"completion" if n_outgo <= 1 else "completions"}, which {impact} the case risk score by {sum_abs_shap * 100:.2f}%.'
            return des

        if n_transact <= 0:  # invalid transaction
            sum_abs_transact_shap = abs(transact_shap) + abs(risk_level_shap)
            des += f'The total transaction is 0 DKK from {countr_risk_level} {impact5} the case risk score by {sum_abs_transact_shap * 100:.2f}%. '
            return des

    # This template is to generate description which is related to observation's portrait
    def cust_resi_country_temp(self, index: int) -> str:
        ori_row = self.ori.iloc[index]
        shap_row = self.shap.iloc[index]

        des = ''

        cust_type = ori_row['Custtype']
        resi_country = ori_row['CustAddressCountryId']
        YearsInDB = ori_row.loc['YearsInDB']
        NumberLinkedCustomers = ori_row.loc['NumberLinkedCustomers']

        cust_type_shap = shap_row['Custtype']
        resi_country_shap = shap_row['CustAddressCountryId']
        bcnf_type_shap = shap_row['bcnf_type_Name']
        sum_shap1 = cust_type_shap + resi_country_shap + bcnf_type_shap

        YearsInDB_shap = shap_row.loc['YearsInDB']
        NumberLinkedCustomers_shap = shap_row.loc['NumberLinkedCustomers']
        sum_shap2 = YearsInDB_shap + NumberLinkedCustomers_shap

        impact1 = 'increased' if sum_shap1 > 0 else 'decreased' if sum_shap1 < 0 else 'remained constant'
        impact2 = 'raised' if sum_shap2 > 0 else 'reduced' if sum_shap2 < 0 else 'remained unchanged'
        # summation of all relevant features contributions
        sum_abs_shap1 = abs(cust_type_shap) + abs(resi_country_shap) + abs(bcnf_type_shap)
        sum_abs_shap2 = abs(YearsInDB_shap) + abs(NumberLinkedCustomers_shap)

        if cust_type == 'individual':
            des += f'\nBesides that, the customer is of type individual and resides in the {resi_country} {impact1} the risk by {sum_abs_shap1 * 100:.2f}%. \n' \
                   f'Additionally, the customer was linked with {int(NumberLinkedCustomers)} other {"customer" if int(NumberLinkedCustomers) <= 1 else "customers"} ' \
                   f'and has been in the database for {int(YearsInDB)} {"year" if int(YearsInDB) <= 1 else "years"}, \n' \
                   f'which {impact2} the risk score by {sum_abs_shap2 * 100:.2f}%.'

        if cust_type == 'organisation':
            des += f'\nWhilst the customer is of type entity, registered in the {resi_country} {impact1} the risk score by {sum_abs_shap1 * 100:.2f}%. \n' \
                   f'Additionally, the company was linked with {int(NumberLinkedCustomers)} other {"entity" if int(NumberLinkedCustomers) <= 1 else "entities"} ' \
                   f'and has been in the database for {int(YearsInDB)} {"year" if int(YearsInDB) <= 1 else "years"}, \n' \
                   f'which {impact2} the risk score by {sum_abs_shap2 * 100:.2f}%.'

        if cust_type == 'infrequent customer type':
            des += f'\nWhilst the customer is of infrequent type, registered in the {resi_country} {impact1} the risk score by {sum_abs_shap1 * 100:.2}%. \n' \
                   f'Additionally, the infrequent customer was linked with {int(NumberLinkedCustomers)} other {"customer" if int(NumberLinkedCustomers) <= 1 else "customers"} ' \
                   f'and has been in the database for {int(YearsInDB)} {"year" if int(YearsInDB) <= 1 else "years"}, \n' \
                   f'which {impact2} the risk score by {sum_abs_shap2 * 100:.2f}%.'

        return des

    # To join description generated by above two templates for each row
    def summary_generator(self, index: int) -> str:
        description = []
        description.append(self.transact_risk_country_temp(index))
        description.append(self.cust_resi_country_temp(index))

        return ' '.join(description)


# Generate description based on the template defined in Description object for each observation
# And store all description for all observations in 'Description' DataFrame
summary_generator = Description(df1, nor_contrib_df)
text = []
for i in range(len(df1)):
    description = summary_generator.summary_generator(i)
    description = description.replace('\n', '')
    text.append(description)

Description = pd.DataFrame(text, columns=['Description'])
# Description.to_csv('description_text.csv')

# Print some instances for checking
##### Case 1: only income transaction #####
print(summary_generator.summary_generator(14))
print('.....................')
print(summary_generator.summary_generator(1))
print('.....................')

##### Case 2: only transfer transaction #####
print(summary_generator.summary_generator(0))
print('.....................')
print(summary_generator.summary_generator(68247))
print('.....................')

##### Case 3: both income and transfer transactions #####
print(summary_generator.summary_generator(9))
print('.....................')
print(summary_generator.summary_generator(21))
print('.....................')

##### Case 4: no transaction occurs #####
print(summary_generator.summary_generator(46513))
print('.....................')
