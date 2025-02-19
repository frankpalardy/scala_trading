import org.apache.spark.sql.{DataFrame, SparkSession}

case class Asset(
                           symbol: String,
                           date: String,
                           closePrice: Double,
                           timestamps: Seq[Long],
                           prices: Seq[Double],
                           highs: Seq[Double],
                           lows: Seq[Double],
                           volumes: Seq[Int]
                     )

object Asset {
  def toDF(data: Seq[Asset])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    data.toDF()
  }
}