import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.log4j._
object WineTester {
  def main(args: Array[String]) = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder
      .appName("Wine Price Regression")
      .master("local")
      .getOrCreate()

    //We'll define a partial schema with the values we are interested in. For the sake of the example points is a Double
    val schemaStruct = StructType(
      StructField("Avg Area Income", DoubleType) ::
        StructField("Avg Area House Age", DoubleType) ::
        StructField("Avg Area Number of Rooms", DoubleType) ::
        StructField("Avg Area Number of Bedrooms", DoubleType) ::
        StructField("Area Population", DoubleType) ::
        StructField("Price", DoubleType) :: Nil
    )

    //We read the data from the file taking into account there's a header.
    //na.drop() will return rows where all values are non-null.
    val df = spark.read
      .option("header", true)
      .schema(schemaStruct)
      .csv("D:\\repo\\Clean-USA-Housing.csv")
      .na.drop()

    //We'll split the set into training and test data
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    val labelColumn = "Price"

    //We define two StringIndexers for the categorical variables

/*    val countryIndexer = new StringIndexer()
      .setInputCol("country")
      .setOutputCol("countryIndex")*/

    //We define the assembler to collect the columns into a new column with a single vector - "features"
    val assembler = new VectorAssembler()
      .setInputCols(Array("Avg Area Income","Avg Area House Age","Avg Area Number of Rooms","Avg Area Number of Bedrooms","Area Population"))
      .setOutputCol("features")

    //----
    val output =assembler.transform(df).select(labelColumn,"features");
    //----
    //For the regression we'll use the Gradient-boosted tree estimator
    val gbt = new GBTRegressor()
      .setLabelCol(labelColumn)
      .setFeaturesCol("features")
      .setPredictionCol("Predicted " + labelColumn)
      .setMaxIter(50)

    //We define the Array with the stages of the pipeline
    val stages = Array(
      /*countryIndexer,*/
      assembler,
      gbt
    )

    //Construct the pipeline
    val pipeline = new Pipeline().setStages(stages)

    //We fit our DataFrame into the pipeline to generate a model
    val model = pipeline.fit(trainingData)

    //We'll make predictions using the model and the test data
    val predictions = model.transform(testData)
    println("------")
    predictions.show();
    println("==============")
    //This will evaluate the error/deviation of the regression using the Root Mean Squared deviation
    val evaluator = new RegressionEvaluator()
      .setLabelCol(labelColumn)
      .setPredictionCol("Predicted " + labelColumn)
      .setMetricName("rmse")

    //We compute the error using the evaluator
    val error = evaluator.evaluate(predictions)

    println(error)

    var someInput=""
    model.transform(someInput)
    spark.stop()
  }
}
//https://blog.scalac.io/scala-spark-ml.html