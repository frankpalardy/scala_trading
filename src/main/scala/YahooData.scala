
import java.time.{Instant, LocalDate, ZoneId}
import java.time.format.DateTimeFormatter
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClients
import org.apache.http.util.EntityUtils
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.sql.catalyst.dsl.expressions.{DslExpression, doubleToLiteral, longToLiteral}

import scala.collection.Seq
import scala.collection.mutable.ListBuffer
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.Locale
import scala.concurrent.{ExecutionContext, Future}
import scala.jdk.CollectionConverters.IteratorHasAsScala
import scala.math.Fractional.Implicits.infixFractionalOps


object YahooData {

  // Removed unused import: import spark.implicits._

  // Function to fetch historical stock data from Yahoo Finance
  def fetchYahooFinanceData(symbol: String, period1: Long, period2: Long): List[LongStockPrice] = {
    val encodedSymbol = URLEncoder.encode(symbol, StandardCharsets.UTF_8.toString)
    val url = s"https://query1.finance.yahoo.com/v8/finance/chart/$encodedSymbol?period1=$period1&period2=$period2&interval=1m"

    val httpClient = HttpClients.createDefault()
    val httpGet = new HttpGet(url)
    httpGet.setHeader("User-Agent", "Mozilla/5.0")

    val response = httpClient.execute(httpGet)
    val entity = response.getEntity
    val result = EntityUtils.toString(entity)

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    val jsonNode = mapper.readTree(result)
    val chartResult = jsonNode.path("chart").path("result").get(0)
    if (!chartResult.isEmpty) {
      val timestampSeq = chartResult.path("timestamp").elements().asScala.map(_.asLong()).toSeq
      val quote = chartResult.path("indicators").path("quote").get(0)
      val closePricesSeq = quote.path("close").elements().asScala.map(_.asDouble()).toSeq

      // Group data by day
      val groupedData = timestampSeq.zip(closePricesSeq).groupBy { case (timestamp, _) =>
        val instant = Instant.ofEpochSecond(timestamp)
        instant.atZone(ZoneId.of("UTC")).toLocalDate
      }

      // Create LongStockPrice objects for each day
      groupedData.map { case (date, dayData) =>
        val (dayTimestamps, dayClosePrices) = dayData.unzip
        val formattedDate = date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"))
        val closePrice = dayClosePrices.last // Use the last price of the day as the closing price

        LongStockPrice(symbol, formattedDate, closePrice, dayTimestamps, dayClosePrices)
      }.toList.sortBy(_.date)
    } else {
      List.empty[LongStockPrice]
    }
  }
}