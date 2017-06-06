from pyspark import SparkContext, SQLContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorIndexer, \
    CountVectorizer, Word2Vec
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType, ArrayType
from stemming.porter2 import stem
from pyspark.ml.clustering import LDA


def evaluate(predicted, label_col="label", raw_prediction_col="rawPrediction"):
    evaluator = BinaryClassificationEvaluator()
    evaluator.setLabelCol(label_col)
    evaluator.setRawPredictionCol(raw_prediction_col)
    return evaluator.evaluate(predicted)


def tokenize(df, input_col, output_col):
    tokenizer = RegexTokenizer().setInputCol(input_col) \
        .setOutputCol(output_col).setPattern(r"\b[^\d\W]+\b").setGaps(False)
    return tokenizer.transform(df)


def remove_stop_words(df, input_col, output_col):
    remover = StopWordsRemover().setInputCol(input_col).setOutputCol(output_col)
    return remover.transform(df)


def do_stem(df, col_name):
    udf = UserDefinedFunction(lambda x: map(lambda y: stem(y), x), ArrayType(StringType()))
    return df.select(*[udf(column).alias(col_name) if column == col_name else column for column in df.columns])


def tf_idf(df, input_col, output_col, num_features):
    hashing = HashingTF().setInputCol(input_col).setOutputCol("temp").setNumFeatures(num_features)
    hashed = hashing.transform(df)
    idfModel = IDF().setInputCol("temp").setOutputCol(output_col).fit(hashed)
    return idfModel.transform(hashed)


def string_index(df, input_col, output_col):
    indexer = StringIndexer(inputCol=input_col, outputCol=output_col)
    return indexer.fit(df).transform(df)


def vector_index(df, input_col, output_col):
    indexer = VectorIndexer(inputCol=input_col, outputCol=output_col)
    return indexer.fit(df).transform(df)


def count_vector(df, input_col, output_col):
    indexer = CountVectorizer(inputCol=input_col, outputCol=output_col)
    return indexer.fit(df).transform(df)


def word_2_vector(df, input_col, output_col):
    indexer = Word2Vec(inputCol=input_col, outputCol=output_col)
    return indexer.fit(df).transform(df)


def lda(df, num_topics, max_iter):
    lda_model = LDA(k=num_topics, maxIter=max_iter).fit(df)
    return lda_model.transform(df)


def decision_tree(training_data, label_col, features_col):
    dt = DecisionTreeClassifier(labelCol=label_col, featuresCol=features_col)
    return dt.fit(training_data)


sc = SparkContext('local')
sqlContext = SQLContext(sc)
rawDataFrame = sqlContext.read.format("com.databricks.spark.csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("outcome_inner_dog_left_person.csv") \
    .select("dog_SubStatusCode", "DayInLife") \
    .na.drop(subset=["DayInLife"])

# tokenizing
tokenized_df = tokenize(rawDataFrame, "DayInLife", "tokens_raw")
print("Tokenizing done!")

# remove stop words
filtered_df = remove_stop_words(tokenized_df, "tokens_raw", "tokens")
print("Removing stop words done!")

# stemming
stemmed_df = do_stem(filtered_df, "tokens")
print("Stemming done!")
# stemmed_df.show(truncate=False)

# tf-idf with LDA
# tf_idf_transformed = tf_idf(stemmed_df, "tokens", "features", 10)
# tf_idf_lda_transformed = lda(tf_idf_transformed, 20, 10)
# (trainingData, testingData) = tf_idf_lda_transformed.randomSplit([0.8, 0.2])
# dtModel = decision_tree(trainingData, "dog_SubStatusCode", "topicDistribution")
# print("LDA with tf-idf model is done!")


# count-vector with LDA
# count_vectored_transformed = count_vector(stemmed_df, "tokens", "features")
# count_vector_lda_transformed = lda(count_vectored_transformed, 20, 10)
# (trainingData, testingData) = count_vector_lda_transformed.randomSplit([0.8, 0.2])
# dtModel = decision_tree(trainingData, "dog_SubStatusCode", "topicDistribution")
# print("LDA with count-vector model is done!")
#

# word-2-vector
# word_2_vectored_transform = word_2_vector(stemmed_df, "tokens", "features")
# (trainingData, testingData) = word_2_vectored_transform.randomSplit([0.8, 0.2])
# dtModel = decision_tree(trainingData, "dog_SubStatusCode", "features")
# print("word-2-vector model is done!")

# tf-idf without LDA
tf_idf_transformed = tf_idf(filtered_df, "tokens", "features", 10)
(trainingData, testingData) = tf_idf_transformed.randomSplit([0.8, 0.2], seed=11L)
dtModel = decision_tree(trainingData, "dog_SubStatusCode", "features")
print("tf-idf model is done!")

trained = dtModel.transform(trainingData)
predicted = dtModel.transform(testingData)
print("Train Accuracy = %g" % evaluate(trained, "dog_SubStatusCode", "prediction"))
print("Test Accuracy = %g " % evaluate(predicted, "dog_SubStatusCode", "prediction"))
