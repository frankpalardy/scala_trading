import ai.djl.training.Trainer
import ai.djl.training.dataset.Batch
import ai.djl.training.listener.TrainingListener
import ai.djl.training.listener.TrainingListener.BatchData
import ai.djl.ndarray.NDList

class CustomEarlyStoppingListener(patience: Int, minImprovement: Float) extends TrainingListener {
  private var bestLoss = Float.MaxValue
  private var patienceCount = 0
  @volatile private var shouldStop = false

  def getShouldStop: Boolean = shouldStop
  private var currentEpochLoss = 0f
  private var batchCount = 0

  override def onEpoch(trainer: Trainer): Unit = {
    val averageLoss = currentEpochLoss / batchCount
    if (averageLoss < bestLoss - minImprovement) {
      bestLoss = averageLoss
      patienceCount = 0
    } else {
      patienceCount += 1
      if (patienceCount >= patience) {
        shouldStop = true
      }
    }
    currentEpochLoss = 0f
    batchCount = 0
  }

  override def onTrainingBatch(trainer: Trainer, batch: BatchData): Unit = {
    if (shouldStop) {
      throw new RuntimeException("Early stopping triggered")
    }
    // Ensure batch.getPredictions and batch.getLabels are NDList
    val predictions = batch.getPredictions.asInstanceOf[NDList]
    val labels = batch.getLabels.asInstanceOf[NDList]

    // Update loss for the current batch
    val loss = trainer.getLoss.evaluate(predictions, labels).getFloat()
    currentEpochLoss += loss
    batchCount += 1
  }
  override def onTrainingBegin(trainer: Trainer): Unit = {
    // Initialize bestLoss if needed
    bestLoss = Float.MaxValue
  }

  override def onTrainingEnd(trainer: Trainer): Unit = {
    println(s"Training ended with best loss: $bestLoss")
  }

  override def onValidationBatch(trainer: Trainer, batch: BatchData): Unit = {
    // You can implement validation loss tracking here if needed
  }
}