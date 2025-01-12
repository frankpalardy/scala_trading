import org.apache.spark.sql.SparkSession

object StockTradingApp {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession.builder
      .appName("StockPriceApp")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._
    val stockPrices = Seq(
      StockPrice("AAPL", "2025-01-01", 150.0),
      StockPrice("GOOGL", "2025-01-01", 2800.0)
    )

    DatabaseInitializer.createTableAndLoadData(stockPrices)

    val df = StockPrice.toDF(stockPrices)
    df.show()
  }
  val portfolio = new Portfolio

  val apple = Stock("AAPL", 150.0)
  val google = Stock("GOOGL", 2800.0)

  portfolio.buy(apple, 10)
  portfolio.buy(google, 5)

  println(s"Portfolio: ${portfolio.getPortfolio}")
  println(s"Total Value: ${portfolio.getTotalValue}")

  portfolio.sell(apple, 5)
  println(s"Portfolio after selling 5 AAPL: ${portfolio.getPortfolio}")
  println(s"Total Value after selling: ${portfolio.getTotalValue}")

  // Update stock price
  portfolio.updatePrice("AAPL", 160.0)
  println(s"Portfolio after updating AAPL price: ${portfolio.getPortfolio}")
  println(s"Total Value after updating price: ${portfolio.getTotalValue}")

  // Get highest value stock
  portfolio.getHighestValueStock.foreach { case (stock, quantity) =>
    println(s"Highest value stock: ${stock.symbol} with value ${stock.price * quantity}")
  }

  // Get lowest value stock
  portfolio.getLowestValueStock.foreach { case (stock, quantity) =>
    println(s"Lowest value stock: ${stock.symbol} with value ${stock.price * quantity}")
  }
}