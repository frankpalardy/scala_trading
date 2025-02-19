import java.sql.{Connection, DriverManager, SQLException, Statement}

import scala.concurrent.{Future, ExecutionContext}

object PostgresConfig {
  val url = "jdbc:postgresql://localhost:5432/trades"
  val user = "frank"
  val password = "doodle"

  def getConnection: Connection = {
    DriverManager.getConnection(url, user, password)
  }
}

object PostgresDatabaseInitializer {
  def createTableAndLoadData(data: Seq[Asset]): Unit = {
    val connection = PostgresConfig.getConnection
    try {
      val statement = connection.createStatement()
      //createTable(statement)
      insertData(statement, data)
    } catch {
      case e: SQLException =>
        println(s"Database error: ${e.getMessage}")
      case e: Exception =>
        println(s"Unexpected error: ${e.getMessage}")
    } finally {
      connection.close()
    }
  }

  private def createTable(statement: Statement): Unit = {
    val createTableSQL =
      """
        |CREATE TABLE IF NOT EXISTS stocks (
        |  symbol VARCHAR(10) NOT NULL,
        |  date DATE NOT NULL,
        |  closePrice DOUBLE PRECISION NOT NULL,
        |  PRIMARY KEY (symbol, date)
        |);
      """.stripMargin
    statement.execute(createTableSQL)
  }

  private def insertData(statement: Statement, data: Seq[Asset]): Unit = {
    data.foreach { stockPrice =>
      val insertSQL =
        s"""
           |INSERT INTO stocks (symbol, date, closePrice)
           |VALUES ('${stockPrice.symbol}', '${stockPrice.date}', ${stockPrice.closePrice})
           |ON CONFLICT (symbol, date) DO NOTHING;
         """.stripMargin
      statement.execute(insertSQL)
    }
  }
}