import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class LinearRegressionPredictor {
  implicit val spark: SparkSession = SparkSession.builder
    .appName("StockPredictor")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  def trainModel(data: Seq[Asset]): LinearRegressionModel = {
    val df = Asset.toDF(data)

    // Convert date to numeric format (Unix timestamp)
    val dfWithNumericDate = df.withColumn("dateNumeric",
      unix_timestamp(col("date"), "yyyy-MM-dd").cast(DoubleType))

    // Convert symbol to numeric using StringIndexer
    val indexer = new StringIndexer()
      .setInputCol("symbol")
      .setOutputCol("symbolIndex")
    val dfIndexed = indexer.fit(dfWithNumericDate).transform(dfWithNumericDate)

    // Explode the prices array into separate rows
    val dfExploded = dfIndexed.select($"*", posexplode($"prices").as(Seq("priceIndex", "price")))

    val assembler = new VectorAssembler()
      .setInputCols(Array("dateNumeric", "symbolIndex", "priceIndex"))
      .setOutputCol("features")

    val featuresDF = assembler.transform(dfExploded)

    val lr = new LinearRegression()
      .setLabelCol("price")
      .setFeaturesCol("features")

    val model = lr.fit(featuresDF)
    model
  }

  def predict(model: LinearRegressionModel, symbol: String, date: String, priceIndex: Int): Double = {
    import spark.implicits._

    // Create a single-row DataFrame with the symbol and date
    val predictionDF = Seq((symbol, date, priceIndex)).toDF("symbol", "date", "priceIndex")

    // Convert the date to a numeric timestamp
    val dfWithNumericDate = predictionDF.withColumn("dateNumeric",
      unix_timestamp(col("date"), "yyyy-MM-dd").cast(DoubleType))

    // Convert symbol to numeric using StringIndexer
    val indexer = new StringIndexer()
      .setInputCol("symbol")
      .setOutputCol("symbolIndex")
    val dfIndexed = indexer.fit(dfWithNumericDate).transform(dfWithNumericDate)

    // Use VectorAssembler on the numeric date and symbol index
    val assembler = new VectorAssembler()
      .setInputCols(Array("dateNumeric", "symbolIndex", "priceIndex"))
      .setOutputCol("features")

    val featuresDF = assembler.transform(dfIndexed)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("prediction")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(model.transform(featuresDF))
    println(s"LinearRegression (RMSE) on test data = $rmse")

    // Make the prediction
    val prediction = model.transform(featuresDF)
    prediction.select("prediction").as[Double].head()

  }
}