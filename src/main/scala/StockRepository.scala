import java.sql.{Connection, ResultSet, SQLException}
import scala.concurrent.{Future, ExecutionContext}
import scala.util.{Failure, Success, Try}

class StockRepository(implicit ec: ExecutionContext) {

  def getHistoricalPrices(symbol: String): Future[Seq[StockPrice]] = Future {
    val connection = DatabaseConfig.getConnection
    try {
      val statement = connection.prepareStatement("SELECT symbol, date, closePrice FROM stocks WHERE symbol = ?")
      statement.setString(1, symbol)
      val resultSet = statement.executeQuery()
      resultSetToStockPrices(resultSet)
    } catch {
      case e: SQLException =>
        println(s"Database error: ${e.getMessage}")
        Seq.empty[StockPrice]
      case e: Exception =>
        println(s"Unexpected error: ${e.getMessage}")
        Seq.empty[StockPrice]
    } finally {
      connection.close()
    }
  }

  def insertStockPrice(stockPrice: StockPrice): Future[Boolean] = Future {
    val connection = DatabaseConfig.getConnection
    try {
      val statement = connection.prepareStatement("INSERT INTO stocks (symbol, date, closePrice) VALUES (?, ?, ?)")
      statement.setString(1, stockPrice.symbol)
      statement.setString(2, stockPrice.date)
      statement.setDouble(3, stockPrice.closePrice)
      statement.executeUpdate() > 0
    } catch {
      case e: SQLException =>
        println(s"Database error: ${e.getMessage}")
        false
      case e: Exception =>
        println(s"Unexpected error: ${e.getMessage}")
        false
    } finally {
      connection.close()
    }
  }

  def insertStockPrices(stockPrices: List[StockPrice]): Future[Boolean] = Future {
    val connection = DatabaseConfig.getConnection
    try {
      val statement = connection.prepareStatement("INSERT INTO stocks (symbol, date, closePrice) VALUES (?, ?, ?)")
      stockPrices.foreach { stockPrice =>
        statement.setString(1, stockPrice.symbol)
        statement.setString(2, stockPrice.date)
        statement.setDouble(3, stockPrice.closePrice)
        statement.addBatch()
      }
      statement.executeBatch()
      true
    } catch {
      case e: SQLException =>
        println(s"Database error: ${e.getMessage}")
        false
      case e: Exception =>
        println(s"Unexpected error: ${e.getMessage}")
        false
    } finally {
      connection.close()
    }
  }

  private def resultSetToStockPrices(resultSet: ResultSet): Seq[StockPrice] = {
    Iterator.continually((resultSet.next(), resultSet)).takeWhile(_._1).map { case (_, rs) =>
      StockPrice(rs.getString("symbol"), rs.getString("date"), rs.getDouble("closePrice"))
    }.toList
  }
}