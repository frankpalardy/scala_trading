import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.training.{DefaultTrainingConfig, GradientCollector, ParameterStore, Trainer}
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.listener.{EarlyStoppingListener, TrainingListener}
import ai.djl.nn.{Block, Parameter, SequentialBlock}
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.index.NDIndex
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.tracker.Tracker
import ai.djl.training.util.ProgressBar

import java.nio.file.Paths
import ai.djl.util.PairList

import java.time.{Instant, LocalDateTime, ZoneId}

object LSTMPredictor {

  val patience = 10
  val minDelta = 0.001 // Minimum change in loss to be considered as improvement
  val numFeatures = 3
  val hiddenSize = 50
  val batchSize = 20       // Changed to divide 500 evenly
  val epochs =30

  // Add class to hold normalization parameters
  case class NormalizationParams(min: Double, max: Double)

  def calculateSequenceLength(stockDataWeek: List[Asset]): Int = {
    val pointsPerDay = 100  // We're taking top 100 volume points per day
    val tradingDays = stockDataWeek.count(day =>
      day.timestamps.indices.exists(i => day.prices(i) > 0)
    )
    pointsPerDay * tradingDays
  }

  def getNormalizationParams(stockDataWeek: List[Asset]): NormalizationParams = {
    val allValues = stockDataWeek.flatMap { day =>
      day.timestamps.indices
        .filter(i => day.prices(i) > 0)
        .flatMap(i => List(day.prices(i), day.highs(i), day.lows(i)))
    }
    NormalizationParams(allValues.min, allValues.max)
  }

  class LSTMModel extends SequentialBlock {
    private val lstm = ai.djl.nn.recurrent.LSTM
      .builder()
      .setNumLayers(2)
      .setStateSize(hiddenSize)
      .optDropRate(0.2f)
      .build()

    private val dense = ai.djl.nn.core.Linear
      .builder()
      .setUnits(2)
      .build()

    override def forward(parameterStore: ParameterStore, inputs: NDList, training: Boolean): NDList = {
      var current = inputs
      current = lstm.forward(parameterStore, current, training)
      val lastTimeStep = current.head().get(new NDIndex(":, -1, :"))
      current = new NDList(lastTimeStep)
      current = dense.forward(parameterStore, current, training)
      current
    }
    add(lstm)
    add(dense)
  }


  def prepareData(stockDataWeek: List[Asset]): (NDArray, NDArray, NormalizationParams) = {
    val manager = NDManager.newBaseManager()
    try {
      // Print daily stats for debugging
      stockDataWeek.sortBy(_.date).foreach { day =>
        val validHighs = day.highs.filter(_ > 0)
        val validLows = day.lows.filter(_ > 0)
        if (validHighs.nonEmpty && validLows.nonEmpty) {
          println(s"Day ${day.date}: High=${validHighs.max}, Low=${validLows.min}, price=${day.closePrice}")
        }
      }

      val normParams = getNormalizationParams(stockDataWeek)
      val sequenceLength = calculateSequenceLength(stockDataWeek)
      // First organize by day
      val dailyData = stockDataWeek.sortBy(_.date).map { day =>
        day.timestamps.indices
          .filter { i =>
            val timestamp = day.timestamps(i)
            val hour = LocalDateTime.ofInstant(
              Instant.ofEpochSecond(timestamp),
              ZoneId.of("America/New_York")
            ).getHour
            hour >= 9 && hour < 16 &&
              day.prices(i) > 0 && day.highs(i) > 0 && day.lows(i) > 0
          }
          .map { i =>
            (day.prices(i), day.highs(i), day.lows(i))
          }
      }

      // Print points per day
      dailyData.zipWithIndex.foreach { case (dayData, i) =>
        println(s"Day ${i+1} has ${dayData.length} valid points")
      }
      // Combine all days into one continuous sequence
      val allData = stockDataWeek.sortBy(_.date).flatMap { day =>
        val dayPoints = day.timestamps.indices
          .filter { i =>
            val timestamp = day.timestamps(i)
            val dt = LocalDateTime.ofInstant(
              Instant.ofEpochSecond(timestamp),
              ZoneId.of("America/New_York")
            )
            // Get indices of top 100 volume points for this day
            val topVolumeIndices = day.volumes.zipWithIndex
              .sortBy(-_._1)
              .take(100)
              .map(_._2)
              .toSet
            val hour = dt.getHour
            ((hour == 16 && dt.getMinute() <= 15) || (hour >= 9 && hour < 16)) &&
              day.prices(i) > 0 && day.highs(i) > 0 && day.lows(i) > 0 &&
              topVolumeIndices.contains(i)  // Only include if it's in top 100 volume
          }
          .map { i =>
            (day.prices(i), day.highs(i), day.lows(i))
          }
        println(s"Day ${day.date} has ${dayPoints.length} points")
        dayPoints
      }
      // Make sure we have enough data
      if (allData.isEmpty) {
        println("No valid data after filtering zeros")
        // Handle empty data case
      }

      println(s"Valid data points after filtering: ${allData.length}")
      // Dynamic sequence length
      val actualLength = allData.length
      val dynamicSequenceLength = actualLength

      println(s"Using sequence length: $dynamicSequenceLength")
      // Get min/max for entire dataset for consistent normalization
      val allValues = allData.flatMap { case (p, h, l) => List(p, h, l) }
      val min = allValues.min
      val max = allValues.max
      // Create sequences using the full dataset with a step size
      val sequences = allData.iterator.sliding(sequenceLength, 10).map { window =>
        val sequence = window.take(dynamicSequenceLength)
        // get next day's high and low
        val targetHigh = stockDataWeek.last.highs.filter(_ > 0).max
        val targetLow = stockDataWeek.last.lows.filter(_ > 0).min
        println(s"Next day target - High: $targetHigh, Low: $targetLow")  // Debug print
        println(s"Window size: ${window.size}")
        println(s"Sequence size: ${sequence.size}")
        println(s"Features array size: ${sequence.map { case (p, h, l) =>
          Array(p.toFloat, h.toFloat, l.toFloat)
        }.flatten.length}")

        val features = sequence.map { case (p, h, l) =>
          Array(
            ((p - min) / (max - min)).toFloat,
            ((h - min) / (max - min)).toFloat,
            ((l - min) / (max - min)).toFloat
          )
        }
        // Predict both high and low
        (features.flatten.toArray, Array((targetHigh - min) / (max - min), (targetLow - min) / (max - min)).map(_.toFloat))
      }.toList

      val features = manager.create(
        sequences.map(_._1).flatten.toArray,
        new Shape(sequences.length, sequenceLength, numFeatures)
      )

      val labels = manager.create(
        sequences.map(_._2).flatten.toArray,
        new Shape(sequences.length, 2)
      )

      println(s"Number of sequences: ${sequences.length}")
      println(s"Feature array length: ${sequences.map(_._1).flatten.length}")
      println(s"Label array length: ${sequences.map(_._2).length}")

      (features, labels, normParams)
    } catch {
      case e: Exception =>
        manager.close()
        throw e
    }
  }

  def train(stockDataWeek: List[Asset]): Model = {

    val model = Model.newInstance("stockPredictor")  // Let DJL choose the default engine
    model.setBlock(new LSTMModel())

    val earlyStoppingListener = new CustomEarlyStoppingListener(5, 0.001f)
    val config = new DefaultTrainingConfig(Loss.l2Loss())
      .optOptimizer(
        Optimizer.adam()
          .optLearningRateTracker(Tracker.fixed(0.001f))
          .build()
      )
      .addEvaluator(new Accuracy()) // Add evaluator for validation
      .addTrainingListeners(
        TrainingListener.Defaults.logging().head,
        earlyStoppingListener
      )
      .optDevices(Array(ai.djl.Device.cpu()))

    val (features, labels, normParams) = prepareData(stockDataWeek)
    try {
      val dataset = new ArrayDataset.Builder()
        .setData(features)
        .optLabels(labels)
        .setSampling(batchSize, true)
        .build()

      val trainer = model.newTrainer(config)
      trainer.initialize(features.getShape())

      for (epoch <- 1 to epochs if !earlyStoppingListener.getShouldStop)
          {
        var epochLoss = 0f
        var batchCount = 0

        val dataIter = dataset.getData(trainer.getManager()).iterator()
        while (dataIter.hasNext()) {
          val batch = dataIter.next()
          try {
            println(s"Batch data shape: ${batch.getData().head().getShape()}")
            println(s"Batch label shape: ${batch.getLabels().head().getShape()}")

            var loss: NDArray = null
            // Use GradientCollector to compute gradients
            val gc = Engine.getInstance().newGradientCollector()
            try {
              val pred = trainer.forward(new NDList(batch.getData().head()))
              println(s"Prediction shape: ${pred.head().getShape()}")
              loss = trainer.getLoss().evaluate(new NDList(pred.head()), new NDList(batch.getLabels().head()))
              // Compute gradients
              gc.backward(loss)
            } finally {
              gc.close()
            }
            // Update parameters
            trainer.step()
            epochLoss += loss.toType(DataType.FLOAT32, false).getFloat()
            batchCount += 1
          } finally {
            batch.close()
          }
        }

        println(f"Epoch $epoch: average loss = ${epochLoss/batchCount}%.6f")
        trainer.notifyListeners(listener => listener.onEpoch(trainer))
      }

      // Save normalization params with model
      model.setProperty("min", normParams.min.toString)
      model.setProperty("max", normParams.max.toString)

      model.save(Paths.get("models"), "stockPredictor")
      model
    } finally {
      features.close()
      labels.close()
    }
  }

  def predict(
               model: Model,
               stockDataWeek: List[Asset]
             ): (Double, Double) = {
    val manager = NDManager.newBaseManager()

    try {
      val (features, labels, normParam) = prepareData(stockDataWeek)  // Use existing prepareData
      val min = normParam.min
      val max = normParam.max

      val input = features  // features should already be properly formatted

      try {
        val predictor = model.newPredictor(new ai.djl.translate.NoopTranslator())
        val prediction = predictor.predict(new NDList(input)).singletonOrThrow()

        val predictions = prediction.toFloatArray
        val predictedHigh = predictions(0) * (max - min) + min
        val predictedLow = predictions(1) * (max - min) + min

        (predictedHigh, predictedLow)
      } finally {
        input.close()
      }
    } finally {
      manager.close()
    }
  }
}