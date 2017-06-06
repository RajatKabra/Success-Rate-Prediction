import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import spark.implicits._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType

val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("mode", "DROPMALFORMED").csv("Book6.csv")

//Converting FoodType text feature 
val t = new RegexTokenizer().setInputCol("FoodType").setOutputCol("words")
val r = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
val htf = new HashingTF().setInputCol("filtered").setOutputCol("features").setNumFeatures(1000)
val idf = new IDF().setInputCol("features").setOutputCol("foodfeatures")

val pipeline = new Pipeline().setStages(Array(t, r, htf, idf))

val model = pipeline.fit(df)

val pipedata = model.transform(df)

val data = pipedata.select('dog_SubStatusCode,'foodfeatures, 'ExerciseType)

//Converting ExerciseType text feature 
val t1 = new RegexTokenizer().setInputCol("ExerciseType").setOutputCol("words")
val r1 = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
val htf1 = new HashingTF().setInputCol("filtered").setOutputCol("features").setNumFeatures(1000)
val idf1 = new IDF().setInputCol("features").setOutputCol("excercisefeatures")

val pipeline1 = new Pipeline().setStages(Array(t1, r1, htf1, idf1))

val model1 = pipeline1.fit(data)

val pipedata1 = model1.transform(data)

val data1 = pipedata1.select('dog_SubStatusCode,'foodfeatures,'excercisefeatures)

//Converting data to Double as it is requirement of DecisionTree API
val data2 = data1.select(data1("dog_SubStatusCode").cast(DoubleType).as("dog_SubStatusCode"), 
data1("excercisefeatures"),
data1("foodfeatures"))

val assembler = new VectorAssembler().setInputCols(Array("foodfeatures","excercisefeatures")).setOutputCol("features")

val output = assembler.transform(data2)
val labeled = output.rdd.map(row => LabeledPoint(row.getAs[Double]("dog_SubStatusCode"),org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))))

val splits = labeled.randomSplit(Array(0.8, 0.2))
val training = splits(0).cache()
val test = splits(1)

// Decision tree Hyper parameters
val numClasses = 2
val impurity = "gini"
val maxDepth = 5
val maxBins = 32
val categoricalFeaturesInfo = Map[Int, Int]()

val dt_model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val scoreAndLabelstrainingData = training.map { point =>
  val score = dt_model.predict(point.features)
  (score, point.label)
}

val scoreAndLabelstestData = test.map { point =>
  val score = dt_model.predict(point.features)
  (score, point.label)
}


var trainingError = scoreAndLabelstrainingData.filter(r => r._1 == r._2).count.toDouble / training.count
printf("Prediction :training data = %.2f%%\n", (100*trainingError))

println("\n");
var testingError = scoreAndLabelstestData.filter(r => r._1 == r._2).count.toDouble / test.count
printf("Prediction :testing data = %.2f%%\n", (100*testingError))

println("\n");

