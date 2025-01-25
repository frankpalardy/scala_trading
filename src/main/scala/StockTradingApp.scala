import org.apache.spark.sql.SparkSession
import java.time.LocalDate
import java.time.format.DateTimeFormatter

object StockTradingApp {
  def main(args: Array[String]): Unit = {

  }
  implicit val spark: SparkSession = SparkSession.builder
    .appName("StockPriceApp")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._
    // Function to generate dates
  def generateDates(start: LocalDate, count: Int): List[String] = {
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
    (0 until count).map(i => start.plusDays(i).format(formatter)).toList
  }
  // Generate more data for AAPL
  val startDate = LocalDate.of(2024, 1, 1)
  val dates = generateDates(startDate, 365)
  val random = new scala.util.Random(42)

  val aaplPrices = dates.map { date =>
    val basePrice = 150.0
    val randomFactor = 0.98 + (random.nextDouble() * 0.04)
    val price = basePrice * randomFactor
    StockPrice("AAPL", date, math.round(price * 100.0) / 100.0)
  }

  val stockPrices = aaplPrices ++ Seq(
    StockPrice("GOOGL", "2025-01-01", 2800.0)
  )

  DatabaseInitializer.createTableAndLoadData(stockPrices)
  val pred = new StockPredictor();
  val model = pred.trainModel(stockPrices)

  val df = StockPrice.toDF(stockPrices)
  val prediction = pred.predict(model, "AAPL:","2025-02-01")
  println(s"Predicted stock price for AAPL on 2025-02-01: $prediction")
  //df.show()

}