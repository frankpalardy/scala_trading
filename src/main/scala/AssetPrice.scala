import org.apache.spark.sql.{DataFrame, SparkSession}

case class AssetPrice(
                           symbol: String,
                           date: String,
                           closePrice: Double,
                           timestamps: Seq[Long],
                           prices: Seq[Double],
                           highs: Seq[Double],
                           lows: Seq[Double],
                           volumes: Seq[Int]
                     )

object AssetPrice {
  def toDF(data: Seq[AssetPrice])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    data.toDF()
  }
}