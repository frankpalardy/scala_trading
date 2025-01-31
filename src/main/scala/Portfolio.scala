import scala.collection.concurrent.TrieMap

class Portfolio {
  private val stocks: TrieMap[String, (Stock, Int)] = TrieMap()

  def buy(stock: Stock, quantity: Int): Unit = {
    stocks.updateWith(stock.symbol) {
      case Some((existingStock, existingQuantity)) => Some((existingStock, existingQuantity + quantity))
      case None => Some((stock, quantity))
    }
  }

  def sell(stock: Stock, quantity: Int): Unit = {
    stocks.updateWith(stock.symbol) {
      case Some((existingStock, existingQuantity)) if existingQuantity >= quantity =>
        if (existingQuantity == quantity) None else Some((existingStock, existingQuantity - quantity))
      case other => other
    }
  }

  def getPortfolio: Map[String, (Stock, Int)] = stocks.toMap

  def getTotalValue: Double = {
    stocks.values.map { case (stock, quantity) => stock.closePrice * quantity }.sum
  }

  def updatePrice(symbol: String, newPrice: Double): Unit = {
    stocks.updateWith(symbol) {
      case Some((stock, quantity)) => Some((stock.copy(closePrice = newPrice), quantity))
      case None => None
    }
  }

  def getHighestValueStock: Option[(Stock, Int)] = {
    stocks.values.maxByOption { case (stock, quantity) => stock.closePrice * quantity }
  }

  def getLowestValueStock: Option[(Stock, Int)] = {
    stocks.values.minByOption { case (stock, quantity) => stock.closePrice * quantity }
  }
}