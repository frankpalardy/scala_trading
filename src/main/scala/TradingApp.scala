import org.apache.spark.sql.SparkSession

import scala.concurrent.ExecutionContext.Implicits.global

object TradingApp {
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

    MongoDatabaseInitializer.createTableAndLoadData(optionData)
    val pred = new LinearRegressionPredictor
    val model = pred.trainModel(stockData)
/* dl version
    val dlmodel = LSTMPredictor.train(optionData)  // First train and get the model
    val targetTimestamp = optionData.last.timestamps.last
    val (predictedHigh, predictedLow) = LSTMPredictor.predict(dlmodel, optionData)
    println(s"Predicted high: $predictedHigh")
    println(s"Predicted low: $predictedLow")

    // linear regression version
    val closingPriceIndex = stockData.head.prices.length - 1
    val df = AssetPrice.toDF(stockData)
    val prediction = pred.predict(model, "SPY", "2025-02-10", closingPriceIndex)
    println(s"Predicted stock price for SPY on 2025-02-10: $prediction")
    df.show() */


    val eurogreeks = OptionGreeks.calculateOptionGreeks(
      marketPrice = 5.57,
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      BusinessDayCalculator.calculateTimeToExpiration("2025-03-15"),
      riskFreeRate = .0454,
      isCall = false
    )

   /* val greeks = AmericanOptionGreeks.calculateAmericanOptionGreeksWithoutMarketPrice(
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
    )*/

    val futureVIX = VolatilityCalculator.estimateFutureVIX(
      currentStockData = stockData,
      futureStockPrice = 500.0, // Your predicted future stock price
      futureDateString = "2025-3-21", // The date for which you're predicting
      vixData.sortBy(_.date)(Ordering[String].reverse).find(_.closePrice > 0).map(_.closePrice).getOrElse(throw new Exception("No valid VIX data found")),
      riskFreeRate = .0454, // divided by 100
    )

    val impliedVol = OptionPriceCalculator.calculateAdjustedVolatility(
      stockData,      // Your L
      vixData.sortBy(_.date)(Ordering[String].reverse).find(_.closePrice > 0).map(_.closePrice).getOrElse(throw new Exception("No valid VIX data found")),
      strikePrice = 580.0,
      BusinessDayCalculator.calculateTimeToExpiration("2025-03-21"),
      isCall = false
    )

    val theoreticalPrice = OptionPriceCalculator.calculateOptionPrice(
      currentPrice = 500,
      strikePrice = 580.0,
      BusinessDayCalculator.calculateTimeToExpiration("2025-03-15"),
      riskFreeRate = 0.0454,
      futureVIX,
      isCall = false
    )

      println(s"Implied Volatility: ${impliedVol}")
    /*  println(s"Delta: ${greeks.delta}")
      println(s"Gamma: ${greeks.gamma}")
      println(s"Theta: ${greeks.theta}")
      println(s"Vega: ${greeks.vega}")
      println(s"Rho: ${greeks.rho}")*/
    println(s"futureVIX: ${futureVIX}")
    println(s"price: ${theoreticalPrice}")

  }
}