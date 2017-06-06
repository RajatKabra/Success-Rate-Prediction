from pyspark import SparkContext, SQLContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType, ArrayType
from stemming.porter2 import stem
from pyspark.ml.clustering import LDA


def evaluationUsingBinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction"):
    evaluator = BinaryClassificationEvaluator()
    evaluator.setLabelCol(labelCol)
    evaluator.setRawPredictionCol(rawPredictionCol)
    return evaluator


sc = SparkContext('local')
sqlContext = SQLContext(sc)
df = sqlContext.read.format("com.databricks.spark.csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("../Data/csv/outcome_inner_dog_left_person.csv").select("dog_SubStatusCode", "DayInLife")
tokenizer = RegexTokenizer().setInputCol("DayInLife") \
    .setOutputCol("tokens_raw").setPattern(r"\b[^\d\W]+\b").setGaps(False)
tokenized = tokenizer.transform(df.na.drop(subset=["DayInLife"]))

remover = StopWordsRemover().setInputCol("tokens_raw").setOutputCol("tokens")
filtered = remover.transform(tokenized)

columnName = "tokens"
udf = UserDefinedFunction(lambda x: map(lambda y: stem(y), x), ArrayType(StringType()))
stemmed = filtered.select(
    *[udf(column).alias(columnName) if column == columnName else column for column in filtered.columns])
stemmed.show(truncate=False)
# filtered.select("tokens").show(truncate=False)

featurizedData = HashingTF().setInputCol("tokens").setOutputCol("rawFeatures").setNumFeatures(40).transform(stemmed)
idfModel = IDF().setInputCol("rawFeatures").setOutputCol("features").fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# rescaledData.select("features").show(truncate=False)
# rescaledData.describe().show()

# rescaledData.select("features").show(truncate=False)
# rescaledData.select("features").show(truncate=False)
# LDA.train(rescaledData.select("features").rdd)

lda = LDA(k=3, maxIter=2)
model = lda.fit(rescaledData)
# model.describeTopics().show(truncate=False)
# size = model.vocabSize();

# topics = model.topicsMatrix()
# for topic in range(3):
#     print("Topic " + str(topic) + ":")
#     for word in range(0, model.vocabSize()):
#         print(" " + str(topics[word][topic]))

dt = DecisionTreeClassifier(labelCol="dog_SubStatusCode", featuresCol="topicDistribution")
transformed = model.transform(rescaledData)

transformed.show()

(trainingData, testingData) = transformed.randomSplit([0.8, 0.2], seed=11L)
dtModel = dt.fit(trainingData)
pred = dtModel.transform(testingData)

pred.show()

evaluator = evaluationUsingBinaryClassificationEvaluator("dog_SubStatusCode", "topicDistribution")

accuracy = evaluator.evaluate(pred)
print("Test Accuracy = %g " % accuracy)

# print(model.describeTopics())



# transformed = model.transform(corpus)
# transformed.show(truncate=False)

# train LDA
# ldaModel = LDA.train(corpus, k=15, maxIterations=100, optimizer="online", docConcentration=2.0, topicConcentration=3.0)

# rescaledData.select("features").write.csv('/Users/thuanbao/Study/csv/output.csv')
