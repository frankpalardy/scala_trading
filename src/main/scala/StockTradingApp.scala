import org.apache.spark.sql.SparkSession

import scala.concurrent.ExecutionContext.Implicits.global

object StockTradingApp {
  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession.builder
      .appName("StockPriceApp")
      .config("spark.master", "local")
      .getOrCreate()
    implicit val stockRepository: StockRepository = new StockRepository()
    // Example usage
    val currentTime = System.currentTimeMillis() / 1000
    val oneMonthAgo = currentTime - (30 * 24 * 60 * 60)
    val yahooData = YahooData.fetchYahooFinanceData("AAPL250207C00250000", oneMonthAgo, currentTime)

    // Use the fetched data for further processing

    DatabaseInitializer.createTableAndLoadData(yahooData)
    val pred = new StockPredictor()
    val model = pred.trainModel(yahooData)

    val df = StockPrice.toDF(yahooData)
    val prediction = pred.predict(model, "AAPL250207C00250000", "2025-02-01")
    println(s"Predicted stock price for AAPL250207C00250000 on 2025-02-01: $prediction")
    df.show()

  }
}