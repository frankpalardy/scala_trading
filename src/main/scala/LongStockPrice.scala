import org.apache.spark.sql.{DataFrame, SparkSession}

case class LongStockPrice(symbol: String, date: String, closePrice: Double, timestamp: Seq[Double], close: Seq[Double])


object LongStockPrice {
  def toDF(data: Seq[LongStockPrice])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    data.toDF()
  }
}