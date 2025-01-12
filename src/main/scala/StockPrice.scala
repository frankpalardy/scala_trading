import java.sql.{Connection, DriverManager, ResultSet, SQLException}
import scala.concurrent.{Future, ExecutionContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

case class StockPrice(symbol: String, date: String, price: Double)

import java.sql.{Connection, DriverManager}

object DatabaseConfig {
  val url = "jdbc:postgresql://localhost:5432/trades"
  val user = "frank"
  val password = "doodle"

  def getConnection: Connection = {
    DriverManager.getConnection(url, user, password)
  }
}

object StockPrice {
  def toDF(data: Seq[StockPrice])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    data.toDF()
  }
}