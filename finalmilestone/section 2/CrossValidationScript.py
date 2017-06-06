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
"dbc_DogBreedDescription",
"Age",
"LeftUnattended",
"Weight"
],[
"GoodAppetite",
"Health",
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
"QuietInCrate",
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
"EarCleaning",
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

data = sqlContext.read.format("com.databricks.spark.csv").option("header",True).option("inferSchema", True).option("delimiter",",").load(path)

# Iterate through each feature set
for i in range(len(feature_sets)):

  print("~~~~~~~~Training on feature set "+str(i+1)+"~~~~~~~~")
  feature_set = data.select(label,*feature_sets[i])
  assembler = VectorAssembler(inputCols=feature_sets[i],outputCol=feature_col)

  (trainingData, testData) = feature_set.randomSplit([0.8, 0.2],split_seed)

  dt = DecisionTreeClassifier(labelCol=label, featuresCol=feature_col)
  pipeline = Pipeline(stages=[assembler, dt])

  paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2,5,7,10,12,15,20,25,30])
             .addGrid(dt.maxBins, [10,20,30,40,50])
             .addGrid(dt.impurity, ["gini","entropy"])
             .build())

  evaluator = BinaryClassificationEvaluator(labelCol=label)
  cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=5,seed=cv_seed)
  cvmodel=cv.fit(trainingData)

# Print CV errors to file
  with open("CrossValidationError"+str(i+1)+".csv","w") as f:
    header = ""
    for key,_ in cvmodel.getEstimatorParamMaps()[0].items():
      header = header + key.name +","
    header = header+"error"
    f.write(header+"\n")
    for (cvError,paramMap) in zip(cvmodel.avgMetrics,cvmodel.getEstimatorParamMaps()):
      line = ""
      for _,value in paramMap.items():
        line = line+str(value)+","
      f.write(line + str(cvError)+"\n")


  train_predictions = cvmodel.transform(trainingData)
  test_predictions = cvmodel.transform(testData)

  evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol=label)
  train_accuracy = evaluator.evaluate(train_predictions)
  test_accuracy = evaluator.evaluate(test_predictions)

  bestModelParamMap = max(zip(cvmodel.avgMetrics,cvmodel.getEstimatorParamMaps()),key= lambda item:item[0])[1]

  print("Best Model: " + str({key.name:value for key,value in bestModelParamMap.items()}))
  print("Train Accuracy = %g " % (train_accuracy))
  print("Test Accuracy = %g " % (test_accuracy))