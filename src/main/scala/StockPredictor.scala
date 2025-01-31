import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class StockPredictor {
  implicit val spark: SparkSession = SparkSession.builder
    .appName("StockPredictor")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  def trainModel(data: Seq[StockPrice]): LinearRegressionModel = {
    val df = StockPrice.toDF(data)

    // Convert date to numeric format (Unix timestamp)
    val dfWithNumericDate = df.withColumn("dateNumeric",
      unix_timestamp(col("date"), "yyyy-MM-dd").cast(DoubleType))

    // Convert symbol to numeric using StringIndexer
    val indexer = new StringIndexer()
      .setInputCol("symbol")
      .setOutputCol("symbolIndex")
    val dfIndexed = indexer.fit(dfWithNumericDate).transform(dfWithNumericDate)

    val assembler = new VectorAssembler()
      .setInputCols(Array("dateNumeric", "symbolIndex"))
      .setOutputCol("features")

    val featuresDF = assembler.transform(dfIndexed)

    val lr = new LinearRegression()
      .setLabelCol("closePrice")
      .setFeaturesCol("features")

    val model = lr.fit(featuresDF)
    model
  }

  def predict(model: LinearRegressionModel, symbol: String, date: String): Double = {
    import spark.implicits._

    // Create a single-row DataFrame with the symbol and date
    val predictionDF = Seq((symbol, date)).toDF("symbol", "date")

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
      .setInputCols(Array("dateNumeric", "symbolIndex"))
      .setOutputCol("features")

    val featuresDF = assembler.transform(dfIndexed)

    // Make the prediction
    val prediction = model.transform(featuresDF)
    prediction.select("prediction").as[Double].head()
  }
}