import pandas as pd
import re


# puppy_df = pd.read_csv("PuppyInfo.csv")

def fix_food_amounts(amounts):
    amounts[amounts.astype(float) > 25] = (amounts.astype(float) / 10000)
    return amounts


# ~~~~~~~~~~~~~~~~Clean the weight~~~~~~~~~~~~~~~~
print("Cleaning weight")
digit_regex = "^(\d+\.?\d*)$"
lb_regex = "(\d+\.?\d*)\s*(?:lb|pound|ibs)"
interval_regex = "(\d+\.?\d*)\s*-\s*(\d+\.?\d*)"
start_regex = "^(\d+\.?\d*)"
approx_regex = "(?:rox\.?|~|about|abt|@|#|around|apx|ely|yl|ext|was|>|\?|te|est)\s*(\d+\.?\d*)"

puppy_df = pd.read_csv("PuppyInfo.csv")
puppy_df["Weight"] = puppy_df["Weight"].str.replace(".*twenty four.*", "24", flags=re.IGNORECASE)
puppy_df["Weight"] = puppy_df["Weight"].str.replace(".*sixty.*", "60", flags=re.IGNORECASE)
puppy_df["Weight"] = puppy_df["Weight"].str.replace(".*thirty.*", "30", flags=re.IGNORECASE)
puppy_df["Weight"] = puppy_df["Weight"].str.replace(".*mid seventies.*", "75", flags=re.IGNORECASE)

clean_df = puppy_df["Weight"].str.extract(digit_regex, expand=False)
clean_df = clean_df.fillna(puppy_df["Weight"].str.extract(lb_regex, flags=re.IGNORECASE, expand=False))
clean_df = clean_df.fillna(puppy_df["Weight"].str.extract(interval_regex, expand=False).astype(float).mean(axis=1))
clean_df = clean_df.fillna(puppy_df["Weight"].str.extract(start_regex, expand=False))
clean_df = clean_df.fillna(puppy_df["Weight"].str.extract(approx_regex, flags=re.IGNORECASE, expand=False))
clean_df = clean_df.fillna(clean_df.astype(float).mean())

puppy_df["Weight"] = clean_df

# ~~~~~~~~~~~~~~~~Clean the Age~~~~~~~~~~~~~~~~
print("Cleaning Age")
wordToNumberMap = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                   "eight": "8", "nine": "9"}
year_regex = "(\d+\.?\d*)[\+\s\-]*(?:(\d)\/(\d))?\s*y"
month_regex = "(\d+\.?\d*)[\+\s\-]*(?:(\d)\/(\d))?\s*m"
week_regex = "(\d+\.?\d*)[\+\s\-]*(?:(\d)\/(\d))?\s*w"
day_regex = "(\d+\.?\d*)[\+\s\-]*(?:(\d)\/(\d))?\s*d"
digit_regex = "^(\d+\.?\d*)$"

puppy_df["Age"] = puppy_df["Age"].str.replace("a half", "1/2", flags=re.IGNORECASE)
puppy_df["Age"] = puppy_df["Age"].str.replace("and", "", flags=re.IGNORECASE)
puppy_df["Age"] = puppy_df["Age"].str.replace("a year", "1 y", flags=re.IGNORECASE)
puppy_df["Age"] = puppy_df["Age"].str.replace("5\+", "5", flags=re.IGNORECASE)
puppy_df["Age"] = puppy_df["Age"].str.replace("i/2", "1/2", flags=re.IGNORECASE)
puppy_df["Age"] = puppy_df["Age"].str.replace("15 months \(1yr3m\)", "15 m", flags=re.IGNORECASE)
puppy_df["Age"] = puppy_df["Age"].str.replace("14 weeks on 21 March, 16 wks on Apr4", "16 w", flags=re.IGNORECASE)
puppy_df["Age"] = puppy_df["Age"].str.replace("41 weeks, approx 9.5 months", "41 w", flags=re.IGNORECASE)

for word, number in wordToNumberMap.items():
    puppy_df["Age"] = puppy_df["Age"].str.replace(word, number, flags=re.IGNORECASE)

year_df = puppy_df["Age"].str.extract(year_regex, flags=re.IGNORECASE, expand=False).astype(float)
year_df = year_df[0].add(year_df[1].divide(year_df[2], fill_value=1), fill_value=0).multiply(365)
month_df = puppy_df["Age"].str.extract(month_regex, flags=re.IGNORECASE, expand=False).astype(float)
month_df = month_df[0].add(month_df[1].divide(month_df[2], fill_value=1), fill_value=0).multiply(30)
week_df = puppy_df["Age"].str.extract(week_regex, flags=re.IGNORECASE, expand=False).astype(float)
week_df = week_df[0].add(week_df[1].divide(week_df[2], fill_value=1), fill_value=0).multiply(7)
day_df = puppy_df["Age"].str.extract(day_regex, flags=re.IGNORECASE, expand=False).astype(float)
day_df = day_df[0].add(day_df[1].divide(day_df[2], fill_value=1), fill_value=0)
digit_df = puppy_df["Age"].str.extract(digit_regex, flags=re.IGNORECASE, expand=False).astype(float)
digit_df[digit_df >= 48] = digit_df[digit_df >= 1.5].multiply(7)
digit_df[(digit_df < 48) & (digit_df >= 1.5)] = digit_df[(digit_df < 48) & (digit_df >= 1.5)].multiply(30)
digit_df[digit_df < 1.5] = digit_df[digit_df < 1.5].multiply(365)

clean_df = year_df.add(month_df, fill_value=0)
clean_df = clean_df.add(week_df, fill_value=0)
clean_df = clean_df.add(day_df, fill_value=0)
clean_df = clean_df.add(digit_df, fill_value=0)

puppy_df["Age"] = clean_df
# ~~~~~~~~~~~~~~~~RaiserState~~~~~~~~~~~~~~~~~~~~
print("Cleaning State")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.strip()
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("[\.`]","")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Connecticut","CT")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Delaware","DE")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Maine","ME",flags=re.IGNORECASE)
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Maryland","MD")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Massachusetts","MD")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("New\s+York","NY")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("New YHork","NY")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("New Hampshire","NH")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("New Jersey","NJ")
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Ohio","OH",flags=re.IGNORECASE)
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Pennsylvania","PA",flags=re.IGNORECASE)
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("^V$","VA",flags=re.IGNORECASE)
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Virginia","VA",flags=re.IGNORECASE)
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.replace("Vermont","VT",flags=re.IGNORECASE)
puppy_df["RaiserState"] = puppy_df["RaiserState"].str.upper()
# ~~~~~~~~~~~~~~~~Clean the FoodAmount~~~~~~~~~~~~~~~~
print("Cleaning FoodAmount")
def fix_food_amount(amount):
    amount_in_float = float(amount)
    if amount_in_float > 25:
        return round(amount_in_float / 10000, 2)
    return amount_in_float

puppy_df.loc[:,"FoodAmount"] = puppy_df["FoodAmount"].apply(fix_food_amount)

# ~~~~~~~~~~~~~~~~Clean the NbrOvernights3Mo~~~~~~~~~~~~~~~~
print("Cleaning NbrOvernights3Mo")
def transform(nbr):
    none_list = ["none", "na", "any", "yet", "zero", "0", "no", "hasn't", "haven't"]
    nbr_in_string = str(nbr).strip()
    for none in none_list:
        if none in nbr_in_string.lower():
            return 0

    words = nbr_in_string.replace(",", "").split(" ")
    for i in range(len(words)):
        # week
        if words[i] in ["week", "weeks"]:
            return words[i - 1] * 7
        # night / day / time
        if words[i] in ["night", "nights", "overnites",
                        "weekends", "weekend",
                        "overnights", "overnight",
                        "days", "day",
                        "time", "times"]:
            return words[i - 1]
        # month
        if words[i] in ["month", "months"]:
            return words[i - 1] * 30

        # hour
        if words[i] in ["hour", "hours"]:
            return 1
        # lots / multiple
        if words[i] in ["lots", "multiple"]:
            return 90

    # assuming the rest is number
    for word in words:
        if word.isdigit():
            return word

    return nbr

puppy_df["NbrOvernights3Mo"] = puppy_df["NbrOvernights3Mo"].str.replace(".*One.*","1",flags=re.IGNORECASE)
puppy_df["NbrOvernights3Mo"] = puppy_df["NbrOvernights3Mo"].str.replace(".*two.*","3",flags=re.IGNORECASE)
puppy_df["NbrOvernights3Mo"] = puppy_df["NbrOvernights3Mo"].str.replace(".*three.*","3",flags=re.IGNORECASE)
puppy_df["NbrOvernights3Mo"] = puppy_df["NbrOvernights3Mo"].str.replace(".*four.*","3",flags=re.IGNORECASE)
puppy_df["NbrOvernights3Mo"] = puppy_df["NbrOvernights3Mo"].str.replace(".*First.*","1",flags=re.IGNORECASE)
puppy_df["NbrOvernights3Mo"] = puppy_df["NbrOvernights3Mo"].str.replace(".*Twice.*","1",flags=re.IGNORECASE)
puppy_df.loc[:,"NbrOvernights3Mo"] = puppy_df["NbrOvernights3Mo"].apply(transform)

# ~~~~~~~~~~~~~~~~Clean the NumberClasses3Months~~~~~~~~~~~~~~~~
print("Cleaning NumberClasses3Months")
def maxInString(text):
    if str(text) == "":
        return 4
    result = re.findall("\d+",str(text))
    if(len(result) == 0) :
       return 4
    r = max(map(int, result))
    if r > 20:
        return 8
    else:
        return r

def replaceAll(text):
    #     p1 = re.compile(".*\(d+) classes.*")
    if re.search("['all','every','None']", str(text).strip(), re.IGNORECASE):  # all|every|None
        return 8
    res = re.findall("(\d+) classes", str(text), re.IGNORECASE)
    if len(res) > 0:
        return max(map(int, res))
    try:
        if int(text) == 0:
            return 4
    except:
        pass
    return maxInString(text)
puppy_df["NumberClasses3Months"] = map(replaceAll,puppy_df["NumberClasses3Months"])
# print puppy_df["NumberClasses3Months"]

# ~~~~~~~~~~~~~~~~Remove duplicate personid/puppy pairs~~~~~~~~~~~~~~~~
print("Removing Duplicates")
# Select the oldest age for each puppy/trainer pair
puppy_df = puppy_df.groupby(["ogr_DogID", "Raiser_psn_PersonID"]).apply(lambda row: row[row['Age'] == row['Age'].max()])
# Select the highest survey ID for each puppy/trainer pair
puppy_df = puppy_df.groupby(["ogr_DogID", "Raiser_psn_PersonID"]).apply(
    lambda row: row[row['SurveyID'] == row['SurveyID'].max()])

# ~~~~~~~~~~~~~~~~Merge Files~~~~~~~~~~~~~~~~
print("Merging Files")
trainer_df = pd.read_csv("TrainerInfo.csv")
outcome_df = pd.read_csv("PuppyTrainerOutcome.csv")

# ~~~~~~~~~~~~~~~~replace the StatusCode~~~~~~~~~~~~~~~~
def replaceStatus(i):
    if i in [23,25,26,27,55,98,99,121,169]:
        return 1
    else:
        return 0

outcome_df.dog_SubStatusCode = list(map(replaceStatus, outcome_df.dog_SubStatusCode))


puppy_df["GeneralComments"] = puppy_df["GeneralComments"].str.replace("\n", " ")
trainer_df["DayInLife"] = trainer_df["DayInLife"].str.replace("\n", " ")

out_df = pd.merge(outcome_df, puppy_df, how='left', left_on=["dog_DogID", "ogr_PersonID"],
                  right_on=["ogr_DogID", "Raiser_psn_PersonID"])
out_df = pd.merge(out_df, trainer_df, how='left', left_on=["ogr_PersonID", "dog_DogID"],
                  right_on=["PersonID", "dog_DogID"])

out_df.to_csv("outcome_left_dog_left_person.csv", index=False)
out_df.to_excel("outcome_left_dog_left_person.xlsx", index=False)

out_df = pd.merge(outcome_df, puppy_df, how='inner', left_on=["dog_DogID", "ogr_PersonID"],
                  right_on=["ogr_DogID", "Raiser_psn_PersonID"])
out_df = pd.merge(out_df, trainer_df, how='left', left_on=["ogr_PersonID", "dog_DogID"],
                  right_on=["PersonID", "dog_DogID"])

out_df.to_csv("outcome_inner_dog_left_person.csv", index=False)
out_df.to_excel("outcome_inner_dog_left_person.xlsx", index=False)
