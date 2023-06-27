logic_dict = {
    (' total amount in transactions', 'average of  of total amount'):'multiple',
    (' total amount in transactions', ''): ''}


feature_cards = {' total amount in transactions': 'total transaction amount',
                 ' total amount received': 'total received amount',
                 ' total amount transfered': 'total transferred amount',

                 'average of  of total amount': 'average transaction amount',
                 'average  of total amount received': 'average received amount',
                 'average  of total amount transfered': 'average transferred amount',

                 'Number of transactions in case': 'number of transaction',
                 'Number of incoming transactions in case': 'number of incoming transactions',
                 'Number of outgoing transactions in case': 'number of outgoing transactions',

                 'Std of total amount received': 'standard deviation of total received amount',
                 'Std of total amount transfered': 'standard deviation of total transferred amount',

                 'max  of amount received': 'maximum received amount',
                 'max  of amount transfered': 'maximum transfer amount',

                 'NumberLinkedCustomers': 'number of linked customers',
                 'YearsInDB': 'years in Danske Bank',

                 'Customer had No-SAR filled before': 'customer did not have suspicious activity report',
                 'Customer had SAR filled before': 'customer had suspicious activity report',

                 'Number of alerts customer had before': 'number of alerts the customer has previously received',
                 'Customer age': 'customer age',

                # 'CASHDEPOCASHDEPO2': 'cash deposit',
                # 'TRANSACTCRBOUTGO1': 'outgoing cross-border transfers',
                # 'TRANSACTCRBOUTGO2': 'outgoing cross-border transfers',
                # 'TRANSACTCRBINCOM1': 'incoming cross-border transfers',
                # 'TRANSACTCRBINCOMY2': 'incoming cross-border transfers',

                'scenario_dbs00200dk': 'row with string "dbs00200dk"',
                'scenario_dbs00205dk': 'row with string "dbs00205dk"',
                'scenario_dbs018b3dk_high': 'row with string "dbs018b3dk_high"',
                'scenario_dbs10008dk': 'row with string "dbs10008dk"',
                'scenario_dbs10018fi_high': 'row with string "dbs10018fi_high"',
                'scenario_dbs10018no_high': 'row with string "dbs10018no_high"',
                'scenario_dbs10018se_high': 'row with string "dbs10018se_high"',
                'scenario_dbs10018se_medium': 'row with string "dbs10018se_medium"',
                'scenario_dbs10419se_high': 'row with string "dbs10419se_high"',
                'scenario_dbs10622cb_low': 'row with string "dbs10622cb_low"',
                'scenario_dbs10628cb_high': 'row with string "dbs10628cb_high"',
                'scenario_dbs10645cb': 'row with string "dbs10645cb"',
                'scenario_dbs10646cb': 'row with string "dbs10646cb"',
                'scenario_dbsctf001dkiaa': 'row with string "dbsctf001dkiaa"',
                'scenario_dbsftf01ddpdk_low': 'row with string "dbsftf01ddpdk_low"',
                'scenario_dbsftf01sw1se': 'row with string "dbsftf01sw1se"',
                'scenario_dbsftf01sw5se': 'row with string "dbsftf01sw5se"',
                'scenario_dbslat002dkpal': 'row with string "dbslat002dkpal"',
                'scenario_dbslat002nopal': 'row with string "dbslat002nopal"',
                'scenario_dbs_ctf_02': 'row with string "dbs_ctf_02"',
                'scenario_dbs_ftf_02': 'row with string "dbs_ftf_02"',
                'scenario_dbs_ftf_03': 'row with string "dbs_ftf_03"',
                'scenario_dbs_hrg_03': 'row with string "dbs_hrg_03"',
                'scenario_dbs_hrg_04': 'row with string "dbs_hrg_04"',
                'scenario_dbs_kyc_crb_01': 'row with string "dbs_kyc_crb_01"',
                'scenario_dbs_kyc_ctf_01': 'row with string "dbs_kyc_ctf_01"',
                'scenario_dbs_kyc_hrg_01': 'row with string "dbs_kyc_hrg_01"',
                'scenario_dbs_kyc_tx_01': 'row with string "dbs_kyc_tx_01"',
                'scenario_dbs_lat_01': 'row with string "dbs_lat_01"',
                'scenario_dbs_lat_02': 'row with string "dbs_lat_02"',
                'scenario_dbs_str_01': 'row with string "dbs_str_01"',
                'scenario_dbs_str_02': 'row with string "dbs_str_02"',
                'scenario_dbs_str_03': 'row with string "dbs_str_03"',
                'scenario_kyc10686no_medium': 'row with string "kyc10686no_medium"',
                'scenario_kyc10686se_medium': 'row with string "kyc10686se_medium"',
                'scenario_kyc10786se_medium': 'row with string "kyc10786se_medium"',
                'scenario_kyc11686se_high': 'row with string "kyc11686se_high"',
                'scenario_kyc11686se_medium': 'row with string "kyc11686se_medium"',
                'scenario_kyc11786se_high': 'row with string "kyc11786se_high"',
                'scenario_kyc888p2dk_low': 'row with string "kyc888p2dk_low"',
                'scenario_kyc888p2se_low': 'row with string "kyc888p2se_low"',
                'scenario_kyc_dap_01': 'row with string "kyc_dap_01"',
                'scenario_kyc_dap_02': 'row with string "kyc_dap_02"',
                'scenario_infrequent_sklearn': 'row with infrequent string',

                'cust_risk_score_high': 'high risk score customer',
                'cust_risk_score_low': 'low risk score customer',
                'cust_risk_score_medium': 'medium risk score customer',
                'cust_risk_score_infrequent_sklearn': 'infrequent risk score customer',

                # 'bcnf_type_Name_legal': '',
                # 'bcnf_type_Name_natural',
                # 'bcnf_type_Name_infrequent_sklearn',


                'IntBusinessCd_466900  ': 'customer industry type 466900',
                'IntBusinessCd_467310  ': 'customer industry type 467310',
                'IntBusinessCd_467600  ': 'customer industry type 467600',
                'IntBusinessCd_641900  ': 'customer industry type 641900',
                'IntBusinessCd_643030  ': 'customer industry type 643030',
                'IntBusinessCd_661900  ': 'customer industry type 661900',
                'IntBusinessCd_841100  ': 'customer industry type 841100',
                'IntBusinessCd_privat  ': 'private customer industry type',
                'IntBusinessCd_none': 'none industry type ',
                'IntBusinessCd_infrequent_sklearn': 'industry type is infrequent',

                 # 'CustAddressCountryId_de',
                 # 'CustAddressCountryId_dk',
                 # 'CustAddressCountryId_fi',
                 # 'CustAddressCountryId_fr',
                 # 'CustAddressCountryId_gb',
                 # 'CustAddressCountryId_nl',
                 # 'CustAddressCountryId_no',
                 # 'CustAddressCountryId_se',
                 # 'CustAddressCountryId_tr',
                 # 'CustAddressCountryId_us',
                 # 'CustAddressCountryId_infrequent_sklearn',

                 # 'CustRiskCountryId_de',
                 # 'CustRiskCountryId_dk',
                 # 'CustRiskCountryId_fi',
                 # 'CustRiskCountryId_fr',
                 # 'CustRiskCountryId_gb',
                 # 'CustRiskCountryId_no',
                 # 'CustRiskCountryId_se',
                 # 'CustRiskCountryId_us',
                 # 'CustRiskCountryId_infrequent_sklearn',

                 # 'CustDomcCountryId_de',
                 # 'CustDomcCountryId_dk',
                 # 'CustDomcCountryId_fi',
                 # 'CustDomcCountryId_fr',
                 # 'CustDomcCountryId_gb',
                 # 'CustDomcCountryId_nl',
                 # 'CustDomcCountryId_no',
                 # 'CustDomcCountryId_se',
                 # 'CustDomcCountryId_tr',
                 # 'CustDomcCountryId_us',
                 # 'CustDomcCountryId_infrequent_sklearn',

                 'CountryAmlRiskClassification_a': 'high risk country(A)',
                 'CountryAmlRiskClassification_b': 'medium risk country(B)',
                 'CountryAmlRiskClassification_c': 'low risk country(C)',
                 'CountryAmlRiskClassification_infrequent_sklearn': 'infrequent country risk classification',

                 'IsEeaCountry_false': 'not from the European Economic Area(EEA)',
                 'IsEeaCountry_true': 'from the European Economic Area(EEA)',
                 'IsEeaCountry_infrequent_sklearn': 'infrequent European Economic Area',

                 'IsEmergingMarketCountry_false': 'not from emerging market country',
                 'IsEmergingMarketCountry_true': 'from emerging market country',
                 'IsEmergingMarketCountry_infrequent_sklearn': 'infrequent emerging market country',

                 'IsEuApplicantCountry_false': 'not application EU country',
                 'IsEuApplicantCountry_infrequent_sklearn': 'infrequent application EU country',

                'IsEuCandidateCountry_false': 'not EU candidate country ',
                'IsEuCandidateCountry_true': 'EU candidate country',
                'IsEuCandidateCountry_infrequent_sklearn': 'infrequent EU candidate country',

                'IsEuCountry_false': 'not EU country',
                'IsEuCountry_true': 'EU country',
                'IsEuCountry_infrequent_sklearn': 'infrequent EU country',

                'IsOecdCountry_false': 'not OECD country',
                'IsOecdCountry_true': ' OECD country',
                'IsOecdCountry_infrequent_sklearn': 'infrequent OECD country',

                'IsTaxHavenCountry_false': 'not tex haven country',
                'IsTaxHavenCountry_true': 'tax haven country',
                'IsTaxHavenCountry_infrequent_sklearn': 'infrequent tax haven country',

                'Customer gender_f': 'female',
                'Customer gender_m': 'males',
                'Customer gender_n': 'neutral gender',
                'Customer gender__unknown': 'unknown gender',

                'Custtype_individ': 'individual customer',
                'Custtype_organisa': 'organisation customer',
                'Custtype_infrequent_sklearn': 'infrequent customer type',

                'prediction': 'the probability of the case being false alert',
                 'y': 'y=0 not criminal activity' 'y=1 criminal activity'
}





features = [' total amount in transactions', ' total amount received',
       ' total amount transfered', 'average of  of total amount',
       'average  of total amount received',
       'average  of total amount transfered',
       'Number of transactions in case',
       'Number of incoming transactions in case',
       'Number of outgoing transactions in case',
       'Std of total amount received', 'Std of total amount transfered',
       'max  of amount received', 'max  of amount transfered',
       'NumberLinkedCustomers', 'YearsInDB',
       'Customer had No-SAR filled before',
       'Customer had SAR filled before',
       'Number of alerts customer had before', 'Customer age',
       'CASHDEPOCASHDEPO2', 'TRANSACTCRBOUTGO1', 'TRANSACTCRBOUTGO2',
       'TRANSACTCRBINCOM1', 'TRANSACTCRBINCOMY2', 'scenario_dbs00200dk',
       'scenario_dbs00205dk', 'scenario_dbs018b3dk_high',
       'scenario_dbs10008dk', 'scenario_dbs10018fi_high',
       'scenario_dbs10018no_high', 'scenario_dbs10018se_high',
       'scenario_dbs10018se_medium', 'scenario_dbs10419se_high',
       'scenario_dbs10622cb_low', 'scenario_dbs10628cb_high',
       'scenario_dbs10645cb', 'scenario_dbs10646cb',
       'scenario_dbsctf001dkiaa', 'scenario_dbsftf01ddpdk_low',
       'scenario_dbsftf01sw1se', 'scenario_dbsftf01sw5se',
       'scenario_dbslat002dkpal', 'scenario_dbslat002nopal',
       'scenario_dbs_ctf_02', 'scenario_dbs_ftf_02',
       'scenario_dbs_ftf_03', 'scenario_dbs_hrg_03',
       'scenario_dbs_hrg_04', 'scenario_dbs_kyc_crb_01',
       'scenario_dbs_kyc_ctf_01', 'scenario_dbs_kyc_hrg_01',
       'scenario_dbs_kyc_tx_01', 'scenario_dbs_lat_01',
       'scenario_dbs_lat_02', 'scenario_dbs_str_01',
       'scenario_dbs_str_02', 'scenario_dbs_str_03',
       'scenario_kyc10686no_medium', 'scenario_kyc10686se_medium',
       'scenario_kyc10786se_medium', 'scenario_kyc11686se_high',
       'scenario_kyc11686se_medium', 'scenario_kyc11786se_high',
       'scenario_kyc888p2dk_low', 'scenario_kyc888p2se_low',
       'scenario_kyc_dap_01', 'scenario_kyc_dap_02',
       'scenario_infrequent_sklearn', 'cust_risk_score_high',
       'cust_risk_score_low', 'cust_risk_score_medium',
       'cust_risk_score_infrequent_sklearn', 'bcnf_type_Name_legal',
       'bcnf_type_Name_natural', 'bcnf_type_Name_infrequent_sklearn',
       'IntBusinessCd_466900  ', 'IntBusinessCd_467310  ',
       'IntBusinessCd_467600  ', 'IntBusinessCd_641900  ',
       'IntBusinessCd_643030  ', 'IntBusinessCd_661900  ',
       'IntBusinessCd_841100  ', 'IntBusinessCd_privat  ',
       'IntBusinessCd_none', 'IntBusinessCd_infrequent_sklearn',
       'CustAddressCountryId_de', 'CustAddressCountryId_dk',
       'CustAddressCountryId_fi', 'CustAddressCountryId_fr',
       'CustAddressCountryId_gb', 'CustAddressCountryId_nl',
       'CustAddressCountryId_no', 'CustAddressCountryId_se',
       'CustAddressCountryId_tr', 'CustAddressCountryId_us',
       'CustAddressCountryId_infrequent_sklearn', 'CustRiskCountryId_de',
       'CustRiskCountryId_dk', 'CustRiskCountryId_fi',
       'CustRiskCountryId_fr', 'CustRiskCountryId_gb',
       'CustRiskCountryId_no', 'CustRiskCountryId_se',
       'CustRiskCountryId_us', 'CustRiskCountryId_infrequent_sklearn',
       'CustDomcCountryId_de', 'CustDomcCountryId_dk',
       'CustDomcCountryId_fi', 'CustDomcCountryId_fr',
       'CustDomcCountryId_gb', 'CustDomcCountryId_nl',
       'CustDomcCountryId_no', 'CustDomcCountryId_se',
       'CustDomcCountryId_tr', 'CustDomcCountryId_us',
       'CustDomcCountryId_infrequent_sklearn',
       'CountryAmlRiskClassification_a', 'CountryAmlRiskClassification_b',
       'CountryAmlRiskClassification_c',
       'CountryAmlRiskClassification_infrequent_sklearn',
       'IsEeaCountry_false', 'IsEeaCountry_true',
       'IsEeaCountry_infrequent_sklearn', 'IsEmergingMarketCountry_false',
       'IsEmergingMarketCountry_true',
       'IsEmergingMarketCountry_infrequent_sklearn',
       'IsEuApplicantCountry_false',
       'IsEuApplicantCountry_infrequent_sklearn',
       'IsEuCandidateCountry_false', 'IsEuCandidateCountry_true',
       'IsEuCandidateCountry_infrequent_sklearn', 'IsEuCountry_false',
       'IsEuCountry_true', 'IsEuCountry_infrequent_sklearn',
       'IsOecdCountry_false', 'IsOecdCountry_true',
       'IsOecdCountry_infrequent_sklearn', 'IsTaxHavenCountry_false',
       'IsTaxHavenCountry_true', 'IsTaxHavenCountry_infrequent_sklearn',
       'Customer gender_f', 'Customer gender_m', 'Customer gender_n',
       'Customer gender__unknown', 'Custtype_individ',
       'Custtype_organisa', 'Custtype_infrequent_sklearn']


