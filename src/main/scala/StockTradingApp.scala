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
    val yahooData = YahooData.fetchYahooFinanceData("SPY250321P00575000", oneMonthAgo, currentTime)

    // Use the fetched data for further processing

    DatabaseInitializer.createTableAndLoadData(yahooData)
    val pred = new StockPredictor()
    val model = pred.trainModel(yahooData)

    val df = StockPrice.toDF(yahooData)
    val prediction = pred.predict(model, "SPY250321P00575000", "2025-02-05")
    println(s"Predicted stock price for SPY250321P00575000 on 2025-02-05: $prediction")
    df.show()

    val greeks = OptionGreeks.calculateOptionGreeks(
      marketPrice = 5.57,
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      timeToExpiration = .126,
      riskFreeRate = 4.54,
      isCall = false
    )

    println(s"Implied Volatility: ${greeks.impliedVolatility}")
    println(s"Delta: ${greeks.delta}")
    println(s"Gamma: ${greeks.gamma}")
    println(s"Theta: ${greeks.theta}")
    println(s"Vega: ${greeks.vega}")
    println(s"Rho: ${greeks.rho}")

  }
}