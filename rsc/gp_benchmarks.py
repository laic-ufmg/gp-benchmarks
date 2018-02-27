
# coding: utf-8

# # 1. Imports and Configuration
# Libraries importer:
# * **padas**: Dataframe manipulation
# * **numpy**: Matrix computations
# * **seaborn**: Visualization library
# * **matplotlib.pyplot**: Plotting library
# * **urllib** imports: Dealing with url requests (for data reading)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from urllib.request import Request, urlopen

# Generate a custom diverging colormap (used for seaborn heatmaps)
cmap = sns.diverging_palette(255, 0, as_cmap=True)


# # 2. Recovering the data
# Notice that the datasets are loaded directly from the web. The links for the data are available in our paper and were found from other papers using the dataset. We provide no warranties or conditions of any kind to the availability of these datasets.
# 
# ## 2.1 Dealing with HTTP Error 403
# Some website tries to prevent content scraping. In order to overcome the scraping protection, we used a different http request header, defined inside the function read_csv_from_web

# In[2]:


def read_csv_from_web(url, sep=',', header='infer'):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36' 
                         '(KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36', 
                'Referer': 'https://www.nseindia.com', 
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    url_request  = Request(url, headers = headers)
    return pd.read_csv(urlopen(url_request), sep = sep, header = header)


# ## 2.2 URL for the data
# These links were recovered from the papers and were active on the date we executed this scrip.

# In[3]:


data_urls = {'towerData' : "http://symbolicregression.com/sites/default/files/DataSets/towerData.txt",
             'toxicity' : "http://personal.disco.unimib.it/Vanneschi/toxicity.txt",
             'CCun' : "http://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
             'CCn' : "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
             'DLBCL' : "https://llmpp.nih.gov/DLBCL/DLBCL_patient_data_NEW.txt"
            }


# ## 3.1 Communities and Crime (CCun and CCn)
# These datasets are used in the paper "*Feature Selection to Improve Generalization of Genetic Programming for High-Dimensional Symbolic Regression*". The authors applied some pre-processing to the data before using in their experiemnts (this information was obtained directly from the authors, since it is not present in the paper):
# 1. Discard the first 5 features, since they are non-predictive features.
# 2. From the 18 potential goal/target, we use only the “ViolentCrimesPerPop” and discard the other 17 features (only for the unnormalized version).
# 3. Discard the instances where the target value is missing.
# 4. Use a simple imputation method (set the missing value to be the mean value of the feature) for instances where the feature values are missing,
# 

# In[4]:


ccun_df = read_csv_from_web(data_urls['CCun'], header=None)
ccn_df = read_csv_from_web(data_urls['CCn'], header=None)


# In[5]:


# Column names, obtained from 'http://archive.ics.uci.edu/ml/datasets/communities+and+crime+unnormalized'
ccun_col_names = ["communityname", "state", "countyCode", "communityCode", "fold",
                  "population", "householdsize", "racepctblack", "racePctWhite",
                  "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
                  "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome",
                  "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst",
                  "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", 
                  "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
                  "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",
                  "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
                  "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf",
                  "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv",
                  "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par",
                  "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumKidsBornNeverMar",
                  "PctKidsBornNeverMar", "NumImmig", "PctImmigRecent", "PctImmigRec5",
                  "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
                  "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
                  "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
                  "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
                  "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
                  "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded",
                  "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb",
                  "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "OwnOccQrange",
                  "RentLowQ", "RentMedian", "RentHighQ", "RentQrange", "MedRent",
                  "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg",
                  "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState",
                  "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", 
                  "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop",
                  "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop",
                  "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp",
                  "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
                  "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens",
                  "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr",
                  "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
                  "murders", "murdPerPop", "rapes", "rapesPerPop", "robberies",
                  "robbbPerPop", "assaults", "assaultPerPop", "burglaries", "burglPerPop",
                  "larcenies", "larcPerPop", "autoTheft", "autoTheftPerPop", "arsons",
                  "arsonsPerPop", "ViolentCrimesPerPop", "nonViolPerPop"]

# Column names, obtained from 'http://archive.ics.uci.edu/ml/datasets/communities+and+crime'
ccn_col_names = ["state", "county", "community", "communityname", "fold", "population",
                 "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
                 "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
                 "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
                 "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
                 "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap",
                 "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov",
                 "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
                 "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu",
                 "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
                 "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
                 "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom",
                 "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5",
                 "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
                 "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
                 "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", 
                 "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", 
                 "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", 
                 "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", 
                 "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", 
                 "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", 
                 "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", 
                 "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", 
                 "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", 
                 "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop",
                 "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq",
                 "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop",
                 "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp",
                 "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
                 "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens",
                 "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr",
                 "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
                 "ViolentCrimesPerPop"]


# In[6]:


# Set the name of the columns
ccun_df.columns = ccun_col_names
ccn_df.columns = ccn_col_names

# Discard the first 5 features and the 17 goals we are not interested (for CCun)
col_to_drop  = list(range(5)) + list(range(ccun_df.shape[1] - 18, ccun_df.shape[1] - 2)) + [146]
proc_ccun_df = ccun_df.drop(ccun_df.columns[col_to_drop], axis = 1)

# Discard only the first 5 features (for CCn)
proc_ccn_df = ccn_df.drop(ccn_df.columns[list(range(5))], axis = 1)

# Remove instances where the target value is missing (only for CCun)
proc_ccun_df = proc_ccun_df[proc_ccun_df.ViolentCrimesPerPop != "?"]

# Replace missing values ("?") by NaN
proc_ccun_df = proc_ccun_df.replace('?', np.NaN)
proc_ccn_df = proc_ccn_df.replace('?', np.NaN)

# The values in the columns with ? are treated as string. 
# We have to cast them to numeric values
proc_ccun_df = proc_ccun_df.apply(lambda c : pd.to_numeric(c))
proc_ccn_df = proc_ccn_df.apply(lambda c : pd.to_numeric(c))

# Now we replace all the missing values (now represented as NaN) by
# the mean of the respective column
proc_ccun_df = proc_ccun_df.fillna(proc_ccun_df.mean())
proc_ccn_df = proc_ccn_df.fillna(proc_ccn_df.mean())


# In[7]:


print("Information about CCun\n=======================")
proc_ccun_df.info()
print("\nInformation about CCn\n=======================")
proc_ccn_df.info()


# ### 3.1.2 What we already know
# First, we know that all the attributes are numerical. The UCI ML repository presents a short description of each feature of the dataset:
# - population: population for community: 
# - householdsize: mean people per household 
# - racepctblack: percentage of population that is african american 
# - racePctWhite: percentage of population that is caucasian 
# - racePctAsian: percentage of population that is of asian heritage 
# - racePctHisp: percentage of population that is of hispanic heritage 
# - agePct12t21: percentage of population that is 12-21 in age 
# - agePct12t29: percentage of population that is 12-29 in age 
# - agePct16t24: percentage of population that is 16-24 in age 
# - agePct65up: percentage of population that is 65 and over in age 
# - numbUrban: number of people living in areas classified as urban 
# - pctUrban: percentage of people living in areas classified as urban 
# - medIncome: median household income 
# - pctWWage: percentage of households with wage or salary income in 1989 
# - pctWFarmSelf: percentage of households with farm or self employment income in 1989 
# - pctWInvInc: percentage of households with investment / rent income in 1989 
# - pctWSocSec: percentage of households with social security income in 1989 
# - pctWPubAsst: percentage of households with public assistance income in 1989 
# - pctWRetire: percentage of households with retirement income in 1989 
# - medFamInc: median family income (differs from household income for non-family households) 
# - perCapInc: per capita income 
# - whitePerCap: per capita income for caucasians 
# - blackPerCap: per capita income for african americans 
# - indianPerCap: per capita income for native americans 
# - AsianPerCap: per capita income for people with asian heritage 
# - OtherPerCap: per capita income for people with 'other' heritage 
# - HispPerCap: per capita income for people with hispanic heritage 
# - NumUnderPov: number of people under the poverty level 
# - PctPopUnderPov: percentage of people under the poverty level 
# - PctLess9thGrade: percentage of people 25 and over with less than a 9th grade education 
# - PctNotHSGrad: percentage of people 25 and over that are not high school graduates 
# - PctBSorMore: percentage of people 25 and over with a bachelors degree or higher education 
# - PctUnemployed: percentage of people 16 and over, in the labor force, and unemployed 
# - PctEmploy: percentage of people 16 and over who are employed 
# - PctEmplManu: percentage of people 16 and over who are employed in manufacturing 
# - PctEmplProfServ: percentage of people 16 and over who are employed in professional services 
# - PctOccupManu: percentage of people 16 and over who are employed in manufacturing 
# - PctOccupMgmtProf: percentage of people 16 and over who are employed in management or professional occupations 
# - MalePctDivorce: percentage of males who are divorced 
# - MalePctNevMarr: percentage of males who have never married 
# - FemalePctDiv: percentage of females who are divorced 
# - TotalPctDiv: percentage of population who are divorced 
# - PersPerFam: mean number of people per family 
# - PctFam2Par: percentage of families (with kids) that are headed by two parents 
# - PctKids2Par: percentage of kids in family housing with two parents 
# - PctYoungKids2Par: percent of kids 4 and under in two parent households 
# - PctTeen2Par: percent of kids age 12-17 in two parent households 
# - PctWorkMomYoungKids: percentage of moms of kids 6 and under in labor force 
# - PctWorkMom: percentage of moms of kids under 18 in labor force 
# - NumIlleg: number of kids born to never married 
# - PctIlleg: percentage of kids born to never married 
# - NumImmig: total number of people known to be foreign born 
# - PctImmigRecent: percentage of _immigrants_ who immigated within last 3 years 
# - PctImmigRec5: percentage of _immigrants_ who immigated within last 5 years 
# - PctImmigRec8: percentage of _immigrants_ who immigated within last 8 years 
# - PctImmigRec10: percentage of _immigrants_ who immigated within last 10 years 
# - PctRecentImmig: percent of _population_ who have immigrated within the last 3 years 
# - PctRecImmig5: percent of _population_ who have immigrated within the last 5 years 
# - PctRecImmig8: percent of _population_ who have immigrated within the last 8 years 
# - PctRecImmig10: percent of _population_ who have immigrated within the last 10 years 
# - PctSpeakEnglOnly: percent of people who speak only English 
# - PctNotSpeakEnglWell: percent of people who do not speak English well 
# - PctLargHouseFam: percent of family households that are large (6 or more) 
# - PctLargHouseOccup: percent of all occupied households that are large (6 or more people) 
# - PersPerOccupHous: mean persons per household 
# - PersPerOwnOccHous: mean persons per owner occupied household 
# - PersPerRentOccHous: mean persons per rental household 
# - PctPersOwnOccup: percent of people in owner occupied households 
# - PctPersDenseHous: percent of persons in dense housing (more than 1 person per room) 
# - PctHousLess3BR: percent of housing units with less than 3 bedrooms 
# - MedNumBR: median number of bedrooms 
# - HousVacant: number of vacant households 
# - PctHousOccup: percent of housing occupied 
# - PctHousOwnOcc: percent of households owner occupied 
# - PctVacantBoarded: percent of vacant housing that is boarded up 
# - PctVacMore6Mos: percent of vacant housing that has been vacant more than 6 months 
# - MedYrHousBuilt: median year housing units built 
# - PctHousNoPhone: percent of occupied housing units without phone (in 1990, this was rare!) 
# - PctWOFullPlumb: percent of housing without complete plumbing facilities 
# - OwnOccLowQuart: owner occupied housing - lower quartile value 
# - OwnOccMedVal: owner occupied housing - median value 
# - OwnOccHiQuart: owner occupied housing - upper quartile value 
# - RentLowQ: rental housing - lower quartile rent 
# - RentMedian: rental housing - median rent (Census variable H32B from file STF1A) 
# - RentHighQ: rental housing - upper quartile rent 
# - MedRent: median gross rent (Census variable H43A from file STF3A - includes utilities) 
# - MedRentPctHousInc: median gross rent as a percentage of household income 
# - MedOwnCostPctInc: median owners cost as a percentage of household income - for owners with a mortgage 
# - MedOwnCostPctIncNoMtg: median owners cost as a percentage of household income - for owners without a mortgage 
# - NumInShelters: number of people in homeless shelters 
# - NumStreet: number of homeless people counted in the street 
# - PctForeignBorn: percent of people foreign born 
# - PctBornSameState: percent of people born in the same state as currently living 
# - PctSameHouse85: percent of people living in the same house as in 1985 (5 years before) 
# - PctSameCity85: percent of people living in the same city as in 1985 (5 years before) 
# - PctSameState85: percent of people living in the same state as in 1985 (5 years before) 
# - LemasSwornFT: number of sworn full time police officers 
# - LemasSwFTPerPop: sworn full time police officers per 100K population 
# - LemasSwFTFieldOps: number of sworn full time police officers in field operations (on the street as opposed to administrative etc) 
# - LemasSwFTFieldPerPop: sworn full time police officers in field operations (on the street as opposed to administrative etc) per 100K population 
# - LemasTotalReq: total requests for police 
# - LemasTotReqPerPop: total requests for police per 100K popuation 
# - PolicReqPerOffic: total requests for police per police officer 
# - PolicPerPop: police officers per 100K population 
# - RacialMatchCommPol: a measure of the racial match between the community and the police force. High values indicate proportions in community and police force are similar 
# - PctPolicWhite: percent of police that are caucasian 
# - PctPolicBlack: percent of police that are african american 
# - PctPolicHisp: percent of police that are hispanic 
# - PctPolicAsian: percent of police that are asian 
# - PctPolicMinor: percent of police that are minority of any kind 
# - OfficAssgnDrugUnits: number of officers assigned to special drug units 
# - NumKindsDrugsSeiz: number of different kinds of drugs seized 
# - PolicAveOTWorked: police average overtime worked 
# - LandArea: land area in square miles 
# - PopDens: population density in persons per square mile 
# - PctUsePubTrans: percent of people using public transit for commuting 
# - PolicCars: number of police cars 
# - PolicOperBudg: police operating budget 
# - LemasPctPolicOnPatr: percent of sworn full time police officers on patrol 
# - LemasGangUnitDeploy: gang unit deployed (numeric, **but actually ordinal** - 0 means NO, 1 means YES, 0.5 means Part Time)
# - LemasPctOfficDrugUn: percent of officers assigned to drug units 
# - PolicBudgPerPop: police operating budget per population 
# - ViolentCrimesPerPop: total number of violent crimes per 100K popuation  (target attribute)
# 
# Basically we are trying to predict the number of violent crimes per 100k population for different communities according to data regarding the population living in the communities (salary, age, language spoken), police activity, public transport usage. Let's first check the target attribute:

# In[8]:


ccn_str = str(proc_ccn_df.ViolentCrimesPerPop.describe()).split("\n")
ccun_str = str(proc_ccun_df.ViolentCrimesPerPop.describe()).split("\n")

# Formatting string used to print the information side by side
fmt = '{:<50}{}'

print(fmt.format('======= CCun ========', '======= CCn ======='))
for (ccun_i, ccn_i) in zip(ccun_str, ccn_str):
    print(fmt.format(ccun_i, ccn_i))


# In[9]:


fig, ax =plt.subplots(1,2, figsize=(16,6))

sns.distplot(proc_ccun_df['ViolentCrimesPerPop'], ax=ax[0]).set_title('CCun');
sns.distplot(proc_ccn_df['ViolentCrimesPerPop'], ax=ax[1]).set_title('CCn');


# We can see from the distribution of the target attribute that CCn is a little bit different from CCun. Actually, this difference can be explained by the normalization of values out of the the range [-3\*std, 3\*std] to 0 and 1 (see the UCI page: http://archive.ics.uci.edu/ml/datasets/communities+and+crime).
# We can notice three aspects about both datasets:
# 
# - Deviate from the normal distribution.
# - Have positive skewness.
# - Show kurtosis.
# 
# Let's check the values:

# In[10]:


fmt = '{:<15}{:<20}{}'
print(fmt.format('', '====== CCun ======', '====== CCn ====='))
print(fmt.format("Skewness:", proc_ccun_df['ViolentCrimesPerPop'].skew(), 
                 proc_ccn_df['ViolentCrimesPerPop'].skew()))

print(fmt.format("Kurtosis:", proc_ccun_df['ViolentCrimesPerPop'].kurt(), 
                 proc_ccn_df['ViolentCrimesPerPop'].kurt()))


# The difference in the value reflects the normalization process apoted for values out of the range [-3\*std, 3\*std].

# ### 3.1.3 Looking for insights
# The correlation matrix can give us some insights about the relations between the features and the response variable. Following, we present a heatmap for the pearson correlation between the features of the dataset.

# In[11]:


corr_mat_ccun = proc_ccun_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_mat_ccun, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 23))

# Use white background for the heatmap
sns.set_style("white")

# Plot a heatmap for the correlation matrix, using only the lower triangle
hm = sns.heatmap(corr_mat_ccun, square=True, mask = mask, cmap = cmap, 
                 cbar_kws={"orientation": "horizontal"}, # Using a horizontal color bar
                 xticklabels=corr_mat_ccun.columns.values, # Using all the labels
                 yticklabels=corr_mat_ccun.columns.values);  
    
# Avoiding overlapped labels
hm.tick_params(labelsize=9)
hm.set_title('CCun heatmap');


# In[12]:


corr_mat_ccn = proc_ccn_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_mat_ccn, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 23))

# Use white background for the heatmap
sns.set_style("white")

# Plot a heatmap for the correlation matrix, using only the lower triangle
hm = sns.heatmap(corr_mat_ccn, square=True, mask = mask, cmap = cmap, 
                 cbar_kws={"orientation": "horizontal"}, # Using a horizontal color bar
                 xticklabels=corr_mat_ccn.columns.values, # Using all the labels
                 yticklabels=corr_mat_ccn.columns.values);  
    
# Avoiding overlapped labels
hm.tick_params(labelsize=9)
hm.set_title('CCn heatmap');


# We can see some similarities between CCun and CCn. However, due to the number of features, the heatmap is a little bit messy. However, we can see some features with very high postive (red colors) and negative (blue colors) correlation. Next we show the pair of features with correlation less than -0.6 and greater than 0.6.

# In[20]:


def get_corr_out_range(corr_mat, left, right):
    """Get the values out of the range (left, right)---values <= left or values >= left
    
    Parameters
    ==========
    corr_mat: Pandas.DataFrame
        Correlation matrix
    left: double
        Upper bound of the left interval. Values have to be in (-inf, left]
    right: double
        Lower bound of the right interval. Values have to be in [right, inf)
    """
    # Generates a matrix with True in the upper diagonal and False otherwise
    aux = np.triu(np.ones(corr_mat.shape)).astype(bool)
    aux = corr_mat.mask(aux) # Replace the values indexed by aux by NaN
    aux = aux.stack() # Select only non-NaN values
    # Return the values outside the range (left, right)
    return aux[aux.le(left) | aux.ge(right)]
    
high_corr = get_corr_out_range(corr_mat_ccun, -0.8, 0.8)
fmt = '{:<22}{:<20}{:>5}'
print(fmt.format("Feature 1", "Feature 2", "Corr."))
print("====================================================")
fmt = '{:<22}{:<21}{:>7.4f}'
for i, row in high_corr.iteritems():
    print(fmt.format(i[0], i[1], row))


# We can see some high correlations easy to explain by the complementary nature of the feature:
# * **Distribution of the population in groups**: *racePctWhite* (percentage of population that is caucasian) and *racepctblack* (percentage of population that is african american) have high negative correlation. Possibly these two ethnic groups are the largest ones. Thus, by increasing the percentage of one populatin the percentage of the other group is reduced.
# * **Shared intervals**: *agePctXtY* is the percentage of population with age in \[X,Y\]. Thus, shared intervals have high correlation---e.g., *agePct12t21* and *agePct12t29*. The same occur with the degree of instruction *PctNotHSGrad*-*PctLess9thGrade*.
# * **Most of the population lives in areas classified as urban**: Possible explanation for *numbUrban* and *population* correlation.
# * **Elderly people has no wage and receive social security**: negative cor. for *agePct65up*-*pctWWage* and positive cor. for *agePct65up*-*pctWSocSec*.
# * **Related variables**: *medFamInc*-*medIncome*-*perCapInc*
# * **Investors are those with spare income**: *medFamInc*-*pctWInvInc*
# * **Caucasians possibly earn more than other ethnic groups**: *whitePerCap*-*medIncome*-*medFamInc*-*perCapInc*
# * **The number of people under the poverty level is related to the size of the (urban) population**: *NumUnderPov*-*population*-*numbUrban*.
# * **Only unemplyeds receive public assistance**: *PctUnemployed*-*pctWPubAsst*
# * **Having a job gives you a salary**: *PctEmploy*-*pctWWage*
# 
# 
# 

# In[14]:


proc_ccun_df[["population", "numbUrban", "pctUrban"]].assign(
    new_pctUrban=100*proc_ccun_df.numbUrban/proc_ccun_df.population).assign(diffNewOld=100*proc_ccun_df.numbUrban/proc_ccun_df.population-proc_ccun_df.pctUrban)

