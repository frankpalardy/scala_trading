import scala.collection.concurrent.TrieMap

class Portfolio {
  private val stocks: TrieMap[String, (Asset, Int)] = TrieMap()

  def buy(stock: Asset, quantity: Int): Unit = {
    stocks.updateWith(stock.symbol) {
      case Some((existingStock, existingQuantity)) => Some((existingStock, existingQuantity + quantity))
      case None => Some((stock, quantity))
    }
  }

  def sell(stock: Asset, quantity: Int): Unit = {
    stocks.updateWith(stock.symbol) {
      case Some((existingStock, existingQuantity)) if existingQuantity >= quantity =>
        if (existingQuantity == quantity) None else Some((existingStock, existingQuantity - quantity))
      case other => other
    }
  }

  def getPortfolio: Map[String, (Asset, Int)] = stocks.toMap

  def getTotalValue: Double = {
    stocks.values.map { case (stock, quantity) => stock.closePrice * quantity }.sum
  }

  def updatePrice(symbol: String, newPrice: Double): Unit = {
    stocks.updateWith(symbol) {
      case Some((stock, quantity)) => Some((stock.copy(closePrice = newPrice), quantity))
      case None => None
    }
  }

  def getHighestValueStock: Option[(Asset, Int)] = {
    stocks.values.maxByOption { case (stock, quantity) => stock.closePrice * quantity }
  }

  def getLowestValueStock: Option[(Asset, Int)] = {
    stocks.values.minByOption { case (stock, quantity) => stock.closePrice * quantity }
  }
}