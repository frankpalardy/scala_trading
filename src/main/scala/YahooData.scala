
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClients
import org.apache.http.util.EntityUtils
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

import scala.collection.mutable.ListBuffer
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import scala.concurrent.{ExecutionContext, Future}
import scala.jdk.CollectionConverters.IteratorHasAsScala


object YahooData {

  // Removed unused import: import spark.implicits._

  // Function to fetch historical stock data from Yahoo Finance
  def fetchYahooFinanceData(symbol: String, period1: Long, period2: Long): List[StockPrice] = {
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
    val timestamps = chartResult.path("timestamp").elements().asScala.map(_.asLong()).toList
    val quote = chartResult.path("indicators").path("quote").get(0)
    val closePrices = quote.path("close").elements().asScala.map(_.asDouble()).toList

    val stockPrices = new ListBuffer[StockPrice]()

    for ((timestamp, closePrice) <- timestamps.zip(closePrices)) {
      if (closePrice > 0) {
        val date = new java.util.Date(timestamp * 1000L)
        val formattedDate = new java.text.SimpleDateFormat("yyyy-MM-dd").format(date)
        stockPrices += StockPrice(symbol, formattedDate, closePrice)
      }
    }

    stockPrices.toList
  }
}