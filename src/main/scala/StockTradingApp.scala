import org.apache.spark.sql.SparkSession

import scala.concurrent.ExecutionContext.Implicits.global

object StockTradingApp {
  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession.builder
      .appName("StockPriceApp")
      .config("spark.master", "local")
      .getOrCreate()
    implicit val stockRepository: PostgresRepository = new PostgresRepository()
    // Example usage
    val currentTime = System.currentTimeMillis() / 1000
    val oneWeekAgo = currentTime - (7 * 24 * 60 * 60)
    val vixData = YahooData.fetchYahooFinanceData("^VIX", oneWeekAgo, currentTime)
    val stockData = YahooData.fetchYahooFinanceData("SPY", oneWeekAgo, currentTime)
    val optionData = YahooData.fetchYahooFinanceData("SPY250321P00580000", oneWeekAgo, currentTime)

    MongoDatabaseInitializer.createTableAndLoadData(stockData)
    val pred = new StockPredictor()
    val model = pred.trainModel(stockData)
// dl version
    val dlmodel = DeepStockPredictor.train(stockData)  // First train and get the model
    val targetTimestamp = stockData.last.timestamps.last
    val dlprediction = DeepStockPredictor.predict(dlmodel, stockData, targetTimestamp)
    println(s"Predicted price: $dlprediction")

    // linear regression version
    val closingPriceIndex = stockData.head.prices.length - 1
    val df = LongStockPrice.toDF(stockData)
    val prediction = pred.predict(model, "SPY", "2025-02-07", closingPriceIndex)
    println(s"Predicted stock price for SPY on 2025-02-07: $prediction")
    df.show()

    val eurogreeks = OptionGreeks.calculateOptionGreeks(
      marketPrice = 5.57,
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      timeToExpiration = .126,
      riskFreeRate = .0454,
      isCall = false
    )

    val greeks = AmericanOptionGreeks.calculateAmericanOptionGreeksWithoutMarketPrice(
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      timeToExpiration = .126,
      riskFreeRate = .0454,
      isCall = false,
      volatility = .184
    )

    val amerGreeks = AmericanOptionGreeks.calculateAmericanOptionGreeks(
      marketPrice = 5.57,
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      timeToExpiration = .126,
      riskFreeRate = .0454, // divided by 100
      isCall = false
    )

    val impliedVol = OptionPriceCalculator.calculateAdjustedVolatility(
      stockData,      // Your L
      vixData.last.closePrice,
      strikePrice = 580.0,
      timeToExpiration = 0.109,
      isCall = false
    )

    val theoreticalPrice = OptionPriceCalculator.calculateOptionPrice(
      currentPrice = 600.77,
      strikePrice = 580.0,
      timeToExpiration = 0.11,
      riskFreeRate = 0.0454,
      impliedVol,
      isCall = false
    )

      println(s"Implied Volatility: ${impliedVol}")
    /*  println(s"Delta: ${greeks.delta}")
      println(s"Gamma: ${greeks.gamma}")
      println(s"Theta: ${greeks.theta}")
      println(s"Vega: ${greeks.vega}")
      println(s"Rho: ${greeks.rho}")*/
    println(s"price: ${theoreticalPrice}")

  }
}