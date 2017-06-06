from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark import SparkContext, SQLContext
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

feature_sets = [["GoodAppetite",
"Relationship_Description",
"Age",
"LeftUnattended",
"Weight"
],[
"GoodAppetite",
"Relationship_Description",
"Age",
"LeftUnattended",
"Weight",
"dbc_DogBreedDescription",
"StaysOnCommand",
"Housemanners",
"CounterSurfingJumpOnDoors",
"NoInappropriateChewing"
],[
"GoodAppetite",
"Relationship_Description",
"Age",
"LeftUnattended",
"Weight",
"dbc_DogBreedDescription",
"StaysOnCommand",
"Housemanners",
"CounterSurfingJumpOnDoors",
"NoInappropriateChewing",
"Health",
"OnFurniture",
"BarksExcessively",
"PlaybitePeople",
"EliminationInHouse"
],[
"GoodAppetite",
"Relationship_Description",
"Age",
"LeftUnattended",
"Weight",
"dbc_DogBreedDescription",
"StaysOnCommand",
"Housemanners",
"CounterSurfingJumpOnDoors",
"NoInappropriateChewing",
"Health",
"OnFurniture",
"BarksExcessively",
"PlaybitePeople",
"EliminationInHouse",
"TrafficFear",
"QuietInCrate",
"JumpOnPeople",
"WalksWellOnLeash",
"RespondsToCommandKennel"
],[
"BehavesWellClass",
"SitsOnCommand",
"Sex",
"EarCleaning",
"NoiseFear"
]]

split_seed = 100000
cv_seed = 100000

sc = SparkContext('local')
sqlContext = SQLContext(sc)

path="outcome_inner_dog_left_person.csv"
feature_col = "features"
label = "dog_SubStatusCode"

bestParams = [{'impurity': 'entropy', 'maxDepth': 7, 'maxBins': 10},
{'impurity': 'entropy', 'maxDepth': 5, 'maxBins': 30},
{'impurity': 'entropy', 'maxDepth': 2, 'maxBins': 10},
{'impurity': 'entropy', 'maxDepth': 2, 'maxBins': 10},
{'impurity': 'gini', 'maxDepth': 30, 'maxBins': 50}]

data = sqlContext.read.format("com.databricks.spark.csv").option("header",True).option("inferSchema", True).option("delimiter",",").load(path)

for i in range(5):
  print("~~~~~~~~Training on feature set "+str(i+1)+"~~~~~~~~")
  feature_set = data.select(label,*feature_sets[i])
  assembler = VectorAssembler(inputCols=feature_sets[i],outputCol=feature_col)

  test_errors=""
  train_errors=""

  for j in range(10):
	  (trainingData, testData) = feature_set.randomSplit([0.8, 0.2],split_seed+j+1)
	  
	  dt = DecisionTreeClassifier(labelCol=label, featuresCol=feature_col, impurity=bestParams[i]["impurity"], maxBins=bestParams[i]["maxBins"], maxDepth=bestParams[i]["maxDepth"])
	  pipeline = Pipeline(stages=[assembler, dt])
	  model = pipeline.fit(trainingData)

	  train_predictions = model.transform(trainingData)
	  test_predictions = model.transform(testData)

	  evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol=label)
	  train_accuracy = evaluator.evaluate(train_predictions)
	  test_accuracy = evaluator.evaluate(test_predictions)

	  train_errors=train_errors+","+str(train_accuracy)
	  test_errors=test_errors+","+str(test_accuracy)

  print("Train Errors: "+train_errors[1:])
  print("Test Errors: "+test_errors[1:])