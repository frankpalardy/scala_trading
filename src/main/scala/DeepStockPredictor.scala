import ai.djl.ndarray.{NDArray, NDManager, NDList}
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.training.{DefaultTrainingConfig, Trainer, ParameterStore}
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.listener.TrainingListener
import ai.djl.nn.{Block, Parameter, AbstractBlock}
import ai.djl.Model
import java.nio.file.Paths
import ai.djl.util.PairList

object DeepStockPredictor {
  val sequenceLength = 390 * 7  // Full week of trading minutes
  val numFeatures = 3      // price, high, low
  val hiddenSize = 50
  val batchSize = 32
  val epochs = 100

  class LSTMModel extends AbstractBlock {
    private val lstm = ai.djl.nn.recurrent.LSTM
      .builder()
      .setNumLayers(2)
      .setStateSize(hiddenSize)
      .setDropRate(0.2)
      .build()

    private val dense = ai.djl.nn.core.Linear
      .builder()
      .setUnits(1)
      .build()

    addChildBlock("lstm", lstm)
    addChildBlock("dense", dense)

    override def forwardInternal(
                                  parameterStore: ParameterStore,
                                  inputs: NDList,
                                  training: Boolean,
                                  params: PairList[String, Object]
                                ): NDList = {
      var current = inputs
      current = lstm.forward(parameterStore, current, training)
      current = dense.forward(parameterStore, current, training)
      current
    }

    override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
      Array(new Shape(inputShapes(0).get(0), 1))
    }

    override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit = {
      lstm.initialize(manager, dataType, inputShapes: _*)
      dense.initialize(manager, dataType, lstm.getOutputShapes(inputShapes.toArray): _*)
    }

    override protected def getDirectParameters(): java.util.Collection[Parameter] = {
      new java.util.ArrayList[Parameter]()
    }

    override def getParameterShape(name: String, inputShapes: Array[Shape]): Shape = {
      throw new UnsupportedOperationException("Not implemented")
    }
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
        sequences.map(_._1).toArray,
        new Shape(sequences.length, sequenceLength, numFeatures)
      )

      val labels = manager.create(
        sequences.map(_._2).toArray,
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
    val model = Model.newInstance("stockPredictor")
    model.setBlock(new LSTMModel())

    val config = new DefaultTrainingConfig(Loss.l2Loss())
      .optOptimizer(Optimizer.adam().optLearningRate(0.001f).build())
      .addTrainingListeners(TrainingListener.Defaults.logging())

    val (features, labels) = prepareData(stockDataWeek)
    try {
      val dataset = new ArrayDataset.Builder()
        .setData(features)
        .setLabels(labels)
        .setSampling(batchSize, true)
        .build()

      val trainer = model.newTrainer(config)
      trainer.initialize(features.getShape())

      for (epoch <- 1 to epochs) {
        var epochLoss = 0f
        var batchCount = 0

        for (batch <- dataset.getData(trainer.getManager())) {
          try {
            val pred = trainer.forward(batch)
            val loss = trainer.backward(pred)
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
    val manager = NDManager.newBaseManager()

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
        val predictor = model.newPredictor()
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

