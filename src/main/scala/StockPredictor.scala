import org.apache.spark.ml.feature.VectorAssembler
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
    val assembler = new VectorAssembler()
      .setInputCols(Array("date"))
      .setOutputCol("features")

    val lr = new LinearRegression()
      .setLabelCol("price")
      .setFeaturesCol("features")

    val model = lr.fit(df)
    model
  }

  def predict(model: LinearRegressionModel, date: String): Double = {
    val df = Seq((date.toDouble)).toDF("date")
    val prediction = model.transform(df)
    prediction.select("prediction").as[Double].head()
  }
}