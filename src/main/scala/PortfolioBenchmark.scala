import scala.collection.concurrent.TrieMap 

object PortfolioBenchmark {
  def measureTime[R](block: => R): (R, Long) = {
    val startTime = System.nanoTime()
    val result = block
    val endTime = System.nanoTime()
    (result, endTime - startTime)
  }

  def main(args: Array[String]): Unit = {
    val portfolio = new Portfolio
    val stocks = (1 to 1000000).map(i => Stock(s"STOCK$i", i.toDouble)).toList

    val (result, duration) = measureTime {
      stocks.foreach(stock => portfolio.buy(stock, 10))
    }

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

    println(s"Execution time: ${duration / 1e6} ms")
    println(s"Total portfolio value: ${portfolio.getTotalValue}")
  }
}