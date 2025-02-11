
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
  def fetchYahooFinanceData(symbol: String, period1: Long, period2: Long): List[AssetPrice] = {
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
      val highPricesSeq = quote.path("high").elements().asScala.map(_.asDouble()).toSeq
      val lowPricesSeq = quote.path("low").elements().asScala.map(_.asDouble()).toSeq

      // Group data by day
      val groupedData = timestampSeq.zip(closePricesSeq).zip(highPricesSeq).zip(lowPricesSeq)
        .map { case (((ts, close), high), low) => (ts, close, high, low) }
        .groupBy { case (timestamp, _, _, _) =>
          val instant = Instant.ofEpochSecond(timestamp)
          instant.atZone(ZoneId.of("UTC")).toLocalDate
        }

      // Create AssetPrice objects for each day
      groupedData.map { case (date, dayData) =>
        val dayTimestamps = dayData.map(_._1)
        val dayClosePrices = dayData.map(_._2)
        val dayHighPrices = dayData.map(_._3)
        val dayLowPrices = dayData.map(_._4)

        val formattedDate = date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"))
        val closePrice = dayClosePrices.last

        AssetPrice(
          symbol = symbol,
          date = formattedDate,
          closePrice = closePrice,
          timestamps = dayTimestamps,
          prices = dayClosePrices,
          highs = dayHighPrices,
          lows = dayLowPrices
        )
      }.toList.sortBy(_.date)
    } else {
      List.empty[AssetPrice]
    }
  }
}