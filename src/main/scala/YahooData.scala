
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
    val url = s"https://query1.finance.yahoo.com/v8/finance/chart/$encodedSymbol?period1=$period1&period2=$period2&interval=1d"

    val httpClient = HttpClients.createDefault()
    val httpGet = new HttpGet(url)

    // Set a User-Agent to avoid 403 errors
    httpGet.setHeader("User-Agent", "Mozilla/5.0")

    val response = httpClient.execute(httpGet)
    val entity = response.getEntity
    val result = EntityUtils.toString(entity)

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    val jsonNode = mapper.readTree(result)
    val chartResult = jsonNode.path("chart").path("result").get(0)
    val timestampSeq = chartResult.path("timestamp").elements().asScala.toSeq.map(_.asDouble())
    val quote = chartResult.path("indicators").path("quote").get(0)
    val closePrices = quote.path("close").elements().asScala.map(_.asDouble()).toSeq

    val stockPrices = for {
      (timestamp, closePrice) <- timestampSeq.zip(closePrices)
      if closePrice > 0.0
    } yield {
      val instant = Instant.ofEpochSecond(timestamp.toLong)
      val formattedDate = DateTimeFormatter.ofPattern("yyyy-MM-dd").withLocale( Locale.US ).withZone( ZoneId.of("UTC")).format(instant)
      LongStockPrice(symbol, formattedDate, closePrice, timestampSeq, closePrices)
    }

    stockPrices.toList
  }
}