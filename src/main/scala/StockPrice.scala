import org.apache.spark.sql.{DataFrame, SparkSession}

case class StockPrice(symbol: String, date: String, closePrice: Double)


object StockPrice {
  def toDF(data: Seq[StockPrice])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    data.toDF()
  }
}