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

    println(s"Execution time: ${duration / 1e6} ms")
    println(s"Total portfolio value: ${portfolio.getTotalValue}")
  }
}