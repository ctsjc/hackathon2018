import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

object WineTester2 {
  def main(args: Array[String]) = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder
      .appName("Email Importance Regression")
      .master("local")
      .getOrCreate()
//to	from	date	rank
    //We'll define a partial schema with the values we are interested in. For the sake of the example points is a Double
    val schemaStruct = StructType(
      StructField("to", StringType) ::
        StructField("from", StringType) ::
        StructField("date", StringType) ::
        StructField("rank", IntegerType) ::Nil
    )

    //We read the data from the file taking into account there's a header.
    //na.drop() will return rows where all values are non-null.
    val df = spark.read
      .option("header", true)
      .schema(schemaStruct)
      .csv( getClass.getResource("email-rank.csv").getPath )
      .na.drop()
//"email-rank.csv"
    //We'll split the set into training and test data
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    val labelColumn = "rank"

    //We define two StringIndexers for the categorical variables


    val toIndexer = new StringIndexer()
      .setInputCol("to")
      .setOutputCol("toIndex1")

    val fromIndexer = new StringIndexer()
      .setInputCol("from")
      .setOutputCol("fromIndex1")

    val dateIndexer = new StringIndexer()
      .setInputCol("date")
      .setOutputCol("dateIndex1")


    //We define the assembler to collect the columns into a new column with a single vector - "features"
    val assembler = new VectorAssembler()
      .setInputCols(Array("toIndex1","fromIndex1","dateIndex1"))
      .setOutputCol("features")

    //----
   // val output =assembler.transform(df).select(labelColumn,"features");
    //----
    //For the regression we'll use the Gradient-boosted tree estimator
    val gbt = new GBTRegressor()
      .setLabelCol(labelColumn)
      .setFeaturesCol("features")
      .setPredictionCol("Predicted " + labelColumn)
      .setMaxIter(50)

    //We define the Array with the stages of the pipeline
    val stages = Array(
      toIndexer,
      fromIndexer,
      dateIndexer,
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
  //model.transform()
    println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+error)

    val df2 = spark.createDataFrame(Seq(
      ("myself"," myteam","today")
    )).toDF("to",
      "from",
      "date")
    println("------ result -----------")
    model.transform(df2).show()
    spark.stop()
  }
}
//https://blog.scalac.io/scala-spark-ml.html
/*
val newList = values.map(Tuple1(_))
val df2 = spark.createDataFrame(newList).toDF("stuff")
*/
