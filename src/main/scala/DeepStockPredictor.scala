import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.training.{DefaultTrainingConfig, ParameterStore, Trainer}
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.listener.TrainingListener
import ai.djl.nn.{Block, Parameter, SequentialBlock}
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.training.tracker.Tracker

import java.nio.file.Paths
import ai.djl.util.PairList

object DeepStockPredictor {

  System.setProperty("ai.djl.default_engine", "PyTorch")
  val sequenceLength = 390 * 7  // Full week of trading minutes
  val numFeatures = 3      // price, high, low
  val hiddenSize = 50
  val batchSize = 32
  val epochs = 100

  class LSTMModel extends SequentialBlock {
    private val lstm = ai.djl.nn.recurrent.LSTM
      .builder()
      .setNumLayers(2)
      .setStateSize(hiddenSize)
      .optDropRate(0.2f)
      .build()

    private val dense = ai.djl.nn.core.Linear
      .builder()
      .setUnits(1)
      .build()

    override def forward(parameterStore: ParameterStore, inputs: NDList, training: Boolean): NDList = {
      var current = inputs
      current = lstm.forward(parameterStore, current, training)
      current = dense.forward(parameterStore, current, training)
      current
    }

    add(lstm)
    add(dense)
  }

  def prepareData(stockDataWeek: List[LongStockPrice]): (NDArray, NDArray) = {
    val manager = NDManager.newBaseManager()

    try {
      val sequences = stockDataWeek.flatMap { day =>
        val data = day.timestamps.indices.map { i =>
          (day.prices(i), day.highs(i), day.lows(i))
        }

        data.sliding(sequenceLength + 1).map { window =>
          val sequence = window.dropRight(1)
          val target = window.last._1

          val allValues = sequence.flatMap { case (p, h, l) => List(p, h, l) }
          val min = allValues.min
          val max = allValues.max

          val normalizedSeq = sequence.map { case (p, h, l) =>
            Array(
              (p - min) / (max - min),
              (h - min) / (max - min),
              (l - min) / (max - min)
            )
          }

          (normalizedSeq.flatten.toArray, Array((target - min) / (max - min)))
        }
      }.toList

      val features = manager.create(
        sequences.map(_._1).flatten.toArray,
        new Shape(sequences.length, sequenceLength, numFeatures)
      )

      val labels = manager.create(
        sequences.map(_._2).flatten.toArray,
        new Shape(sequences.length, 1)
      )

      (features, labels)
    } catch {
      case e: Exception =>
        manager.close()
        throw e
    }
  }

  def train(stockDataWeek: List[LongStockPrice]): Model = {
    val engine = Engine.getInstance()
    val model = Model.newInstance("stockPredictor", "PyTorch")
    model.setBlock(new LSTMModel())

    val config = new DefaultTrainingConfig(Loss.l2Loss())
      .optOptimizer(
        Optimizer.adam()
          .optLearningRateTracker(Tracker.fixed(0.001f))
          .build()
      )
      .addTrainingListeners(TrainingListener.Defaults.logging().head)

    val (features, labels) = prepareData(stockDataWeek)
    try {
      val dataset = new ArrayDataset.Builder()
        .setData(features)
        .optLabels(labels)
        .setSampling(batchSize, true)
        .build()

      val trainer = model.newTrainer(config)
      trainer.initialize(features.getShape())

      for (epoch <- 1 to epochs) {
        var epochLoss = 0f
        var batchCount = 0

        val dataIter = dataset.getData(trainer.getManager()).iterator()
        while (dataIter.hasNext()) {
          val batch = dataIter.next()
          try {
            val pred = trainer.forward(new NDList(batch.getData().head()))
            val loss = trainer.getLoss().evaluate(pred, new NDList(batch.getLabels().head()))
            trainer.step()
            epochLoss += loss.getFloat()
            batchCount += 1
          } finally {
            batch.close()
          }
        }

        println(f"Epoch $epoch: average loss = ${epochLoss/batchCount}%.6f")
        trainer.notifyListeners(listener => listener.onEpoch(trainer))
      }

      model.save(Paths.get("models"), "stockPredictor")
      model
    } finally {
      features.close()
      labels.close()
    }
  }

  def predict(
               model: Model,
               stockDataWeek: List[LongStockPrice],
               targetTimestamp: Long
             ): Double = {
    val manager = NDManager.newBaseManager("PyTorch")

    try {
      val lastDay = stockDataWeek.maxBy(_.timestamps.max)
      val targetIndex = lastDay.timestamps.indexOf(targetTimestamp)

      val sequence = lastDay.timestamps.indices
        .slice(targetIndex - sequenceLength, targetIndex)
        .map { i =>
          Array(
            lastDay.prices(i),
            lastDay.highs(i),
            lastDay.lows(i)
          )
        }

      val allValues = sequence.flatten
      val min = allValues.min
      val max = allValues.max

      val normalizedSeq = sequence.map { values =>
        values.map(v => (v - min) / (max - min))
      }

      val input = manager.create(
        normalizedSeq.flatten.toArray,
        new Shape(1, sequenceLength, numFeatures)
      )

      try {
        val predictor = model.newPredictor(new ai.djl.translate.NoopTranslator())
        val prediction = predictor.predict(new NDList(input)).singletonOrThrow()
        prediction.getFloat() * (max - min) + min
      } finally {
        input.close()
      }
    } finally {
      manager.close()
    }
  }
}