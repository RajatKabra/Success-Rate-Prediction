import pandas as pd
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
file_name = "outcome_inner_dog_left_person.xlsx"
target = "dog_SubStatusCode"
success_codes = [23,25,26,27,55,98,99,121,169]
nominal_columns_of_interest = ["dog_Sex","dbc_DogBreedDescription","dbcc_ColorDescription","Relationship_Description","Color","Sex","GoodAppetite","Region","RaiserState"]
ordinal_columns_of_interest = ["Health","StoolFirm","EnergyLevel","EliminationInCrate","QuietInCrate","RespondsToCommandKennel","NoInappropriateChewing","Housemanners","LeftUnattended","EliminationInHouse","PlaybitePeople","StealsFood","OnFurniture","BarksExcessively","RaidsGarbage","CounterSurfingJumpOnDoors","JumpOnPeople","FriendlyWAnimals","GoodWKids","GoodWStrangers","WalksWellOnLeash","KnowCommandGetBusy","EliminatesOnRoute","ChasingAnimals","TrafficFear","NoiseFear","Stairs","SitsOnCommand","DownOnCommand","StaysOnCommand","ComeOnLeash","ComeOffLeash","CanGivePills","EarCleaning","NailCutting","AttendsClasses","BehavesWellClass","AttendsHomeSwitches"]
interval_columns_of_interest = ["FoodAmount","Weight","Age"]

file_df = pd.read_excel(file_name)

# Create arrays
target_array = file_df[target].values
nominal_array = file_df[nominal_columns_of_interest].astype(str).apply(le.fit_transform).values
temp = nominal_columns_of_interest
temp.insert(0,target)
file_df[temp].astype(str).apply(le.fit_transform).to_excel("temp_nominal.xlsx",index=False)
# for column in ordinal_columns_of_interest:
# 	print(column)
# 	file_df.astype(int)
ordinal_array = file_df[ordinal_columns_of_interest].fillna(0).astype(int).values
interval_array = file_df[interval_columns_of_interest].fillna(0).astype(float).values

print("~~~~~~~~~~Chi2~~~~~~~~~~")
test = SelectKBest(score_func=chi2,k="all")
fit = test.fit(nominal_array, target_array)
# summarize scores
numpy.set_printoptions(precision=3)
for i in range(len(fit.scores_)):
	print(nominal_columns_of_interest[i],fit.scores_[i])

print("~~~~~~~~~~ANOVA~~~~~~~~~~")
test = SelectKBest(score_func=f_classif, k="all")
fit = test.fit(ordinal_array,target_array)
numpy.set_printoptions(precision=3)
for i in range(len(fit.scores_)):
	print(ordinal_columns_of_interest[i],fit.scores_[i])
test = SelectKBest(score_func=f_classif, k="all")
fit = test.fit(interval_array,target_array)
numpy.set_printoptions(precision=3)
for i in range(len(fit.scores_)):
	print(interval_columns_of_interest[i],fit.scores_[i])