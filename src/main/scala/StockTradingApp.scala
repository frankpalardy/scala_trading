import org.apache.spark.sql.SparkSession

import scala.concurrent.ExecutionContext.Implicits.global

object StockTradingApp {
  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession.builder
      .appName("StockPriceApp")
      .config("spark.master", "local")
      .getOrCreate()
    implicit val stockRepository: PostgresRepository = new PostgresRepository()
    // Example usage
    val currentTime = System.currentTimeMillis() / 1000
    val oneWeekAgo = currentTime - (7 * 24 * 60 * 60)
    val vixData = YahooData.fetchYahooFinanceData("^VIX", oneWeekAgo, currentTime)
    val stockData = YahooData.fetchYahooFinanceData("SPY", oneWeekAgo, currentTime)
    val optionData = YahooData.fetchYahooFinanceData("SPY250321P00580000", oneWeekAgo, currentTime)

    MongoDatabaseInitializer.createTableAndLoadData(stockData)
    val pred = new StockPredictor()
    val model = pred.trainModel(stockData)

    import org.tensorflow._
    import org.tensorflow.op.Ops
    import org.tensorflow.op.core.{Placeholder, Variable}
    import org.tensorflow.types.TFloat32
    import java.nio.file.Paths

    object DeepStockPredictor {
      // Model parameters
      val sequenceLength = 60  // Look at last 60 minutes
      val numFeatures = 3      // price, high, low
      val lstmUnits = 50
      val batchSize = 32
      val learningRate = 0.001f

      case class ModelComponents(
                                  input: Placeholder[TFloat32],
                                  target: Placeholder[TFloat32],
                                  output: Output[TFloat32],
                                  loss: Output[TFloat32],
                                  optimizer: Output[TFloat32]
                                )

      case class NormalizedData(
                                 tensor: Tensor[TFloat32],
                                 min: Double,
                                 max: Double
                               )

      def buildModel(graph: Graph): ModelComponents = {
        val tf = Ops.create(graph)

        // Input placeholders
        val input = tf.placeholder(TFloat32.class,
        Placeholder.shape(Shape.of(-1, sequenceLength, numFeatures)))
        val target = tf.placeholder(TFloat32.class,
        Placeholder.shape(Shape.of(-1, 1)))

        // LSTM layer
        val lstmCell = tf.nn.basicLSTMCell(lstmUnits)
        val initialState = lstmCell.zeroState(batchSize, TFloat32.class)

        val lstmOutput = tf.nn.dynamicRNN(
          lstmCell,
          input,
          initialState,
          TFloat32.class
        )

        // Dense layer weights and biases
        val weights = tf.variable(
          tf.random.truncatedNormal(
            tf.constant(Shape.of(lstmUnits, 1).asArray()),
            TFloat32.class
        )
        )
        val biases = tf.variable(tf.zeros(Shape.of(1)))

        // Output layer
        val output = tf.add(
          tf.matmul(lstmOutput.getOutput(), weights),
          biases
        )

        // Loss function (MSE)
        val loss = tf.math.reduceMean(
          tf.math.square(tf.math.sub(output, target))
        )

        // Optimizer
        val optimizer = tf.train.adam(learningRate).minimize(loss)

        ModelComponents(input, target, output, loss, optimizer)
      }

      def prepareData(
                       stockDataWeek: List[LongStockPrice]
                     ): List[NormalizedData] = {
        val allData = stockDataWeek.flatMap { day =>
          day.timestamps.zip(day.prices).zip(day.highs).zip(day.lows).map {
            case (((ts, price), high), low) => (ts, price, high, low)
          }
        }.sortBy(_._1)

        // Create sequences
        val sequences = allData.sliding(sequenceLength + 1).map { window =>
          val sequence = window.dropRight(1)
          val target = window.last._2

          // Normalize data
          val allValues = sequence.flatMap { case (_, p, h, l) => List(p, h, l) }
          val min = allValues.min
          val max = allValues.max

          val normalizedSeq = sequence.map { case (_, p, h, l) =>
            Array(
              (p - min) / (max - min),
              (h - min) / (max - min),
              (l - min) / (max - min)
            )
          }

          val tensor = TFloat32.tensorOf(
            Shape.of(1, sequenceLength, numFeatures),
            normalizedSeq.flatten.map(_.toFloat).toArray
          )

          NormalizedData(tensor, min, max)
        }.toList

        sequences
      }

      def train(
                 stockDataWeek: List[LongStockPrice],
                 epochs: Int = 100,
                 modelPath: String = "models/stock_predictor"
               ): Unit = {
        val graph = new Graph()

        try {
          val model = buildModel(graph)
          val session = new Session(graph)

          try {
            // Initialize variables
            session.runner().addTarget("init").run()

            // Prepare training data
            val trainingData = prepareData(stockDataWeek)

            // Training loop
            for (epoch <- 1 to epochs) {
              var epochLoss = 0.0

              // Process in batches
              trainingData.grouped(batchSize).foreach { batch =>
                val batchTensors = batch.map(_.tensor)
                val batchTargets = batch.map(_.tensor) // Need to create proper targets

                // Run training step
                val loss = session.runner()
                  .feed(model.input, batchTensors.head)
                  .feed(model.target, batchTargets.head)
                  .addTarget(model.optimizer)
                  .fetch(model.loss)
                  .run()
                  .get(0)
                  .asInstanceOf[TFloat32]
                  .getFloat()

                epochLoss += loss
              }

              println(s"Epoch $epoch, Loss: ${epochLoss / trainingData.size}")
            }

            // Save model
            val saver = new Saver(graph)
            saver.save(session, modelPath)

          } finally {
            session.close()
          }
        } finally {
          graph.close()
        }
      }

      def predict(
                   stockDataWeek: List[LongStockPrice],
                   targetTimestamp: Long,
                   modelPath: String = "models/stock_predictor"
                 ): Double = {
        val graph = new Graph()

        try {
          val model = buildModel(graph)
          val session = new Session(graph)

          try {
            // Load model
            val saver = new Saver(graph)
            saver.restore(session, modelPath)

            // Prepare prediction data
            val data = prepareData(stockDataWeek)
            val predictionData = data.last

            // Make prediction
            val prediction = session.runner()
              .feed(model.input, predictionData.tensor)
              .fetch(model.output)
              .run()
              .get(0)
              .asInstanceOf[TFloat32]
              .getFloat()

            // Denormalize prediction
            prediction * (predictionData.max - predictionData.min) + predictionData.min

          } finally {
            session.close()
          }
        } finally {
          graph.close()
        }
      }
    }

      DeepStockPredictor.train(stockData)

      // Make prediction
      val targetTimestamp = stockData.last.timestamps.last
      val dlprediction = DeepStockPredictor.predict(stockData, targetTimestamp)
      println(s"Predicted price: $dlprediction")


    val closingPriceIndex = stockData.head.prices.length - 1
    val df = LongStockPrice.toDF(stockData)
    val prediction = pred.predict(model, "SPY", "2025-02-07", closingPriceIndex)
    println(s"Predicted stock price for SPY on 2025-02-07: $prediction")
    df.show()

    val eurogreeks = OptionGreeks.calculateOptionGreeks(
      marketPrice = 5.57,
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      timeToExpiration = .126,
      riskFreeRate = .0454,
      isCall = false
    )

    val greeks = AmericanOptionGreeks.calculateAmericanOptionGreeksWithoutMarketPrice(
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      timeToExpiration = .126,
      riskFreeRate = .0454,
      isCall = false,
      volatility = .184
    )

    val amerGreeks = AmericanOptionGreeks.calculateAmericanOptionGreeks(
      marketPrice = 5.57,
      underlyingPrice = 598.0,
      strikePrice = 575.0,
      timeToExpiration = .126,
      riskFreeRate = .0454, // divided by 100
      isCall = false
    )

    val impliedVol = OptionPriceCalculator.calculateAdjustedVolatility(
      stockData,      // Your L
      vixData.last.closePrice,
      strikePrice = 580.0,
      timeToExpiration = 0.109,
      isCall = false
    )

    val theoreticalPrice = OptionPriceCalculator.calculateOptionPrice(
      currentPrice = 600.77,
      strikePrice = 580.0,
      timeToExpiration = 0.11,
      riskFreeRate = 0.0454,
      impliedVol,
      isCall = false
    )

      println(s"Implied Volatility: ${impliedVol}")
    /*  println(s"Delta: ${greeks.delta}")
      println(s"Gamma: ${greeks.gamma}")
      println(s"Theta: ${greeks.theta}")
      println(s"Vega: ${greeks.vega}")
      println(s"Rho: ${greeks.rho}")*/
    println(s"price: ${theoreticalPrice}")

  }
}