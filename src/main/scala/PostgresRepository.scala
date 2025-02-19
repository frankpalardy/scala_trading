import java.sql.{Connection, ResultSet, SQLException}
import scala.concurrent.{Future, ExecutionContext}
import scala.util.{Failure, Success, Try}

class PostgresRepository(implicit ec: ExecutionContext) {

  def getHistoricalPrices(symbol: String): Future[Seq[Asset]] = Future {
    val connection = PostgresConfig.getConnection
    try {
      val statement = connection.prepareStatement("SELECT symbol, date, closePrice FROM stocks WHERE symbol = ?")
      statement.setString(1, symbol)
      val resultSet = statement.executeQuery()
      resultSetToStockPrices(resultSet)
    } catch {
      case e: SQLException =>
        println(s"Database error: ${e.getMessage}")
        Seq.empty[Asset]
      case e: Exception =>
        println(s"Unexpected error: ${e.getMessage}")
        Seq.empty[Asset]
    } finally {
      connection.close()
    }
  }

  def insertStockPrice(stockPrice: Asset): Future[Boolean] = Future {
    val connection = PostgresConfig.getConnection
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

  def insertStockPrices(stockPrices: List[Asset]): Future[Boolean] = Future {
    val connection = PostgresConfig.getConnection
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

  private def resultSetToStockPrices(resultSet: ResultSet): Seq[Asset] = {
    Iterator.continually((resultSet.next(), resultSet)).takeWhile(_._1).map { case (_, rs) =>
      Asset(
        symbol = rs.getString("symbol"),
        date = rs.getString("date"),
        closePrice = rs.getDouble("closePrice"),
        timestamps = Seq(rs.getLong("timestamp")),
        prices = Seq(rs.getDouble("closePrice")),
        highs = Seq(rs.getDouble("high")),
        lows = Seq(rs.getDouble("lows")),
        volumes = Seq(rs.getInt("volumes")),
      )
    }.toList
  }
}
