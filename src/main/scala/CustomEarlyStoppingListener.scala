import ai.djl.training.Trainer
import ai.djl.training.dataset.Batch
import ai.djl.training.listener.TrainingListener
import ai.djl.training.listener.TrainingListener.BatchData

class CustomEarlyStoppingListener(patience: Int, minImprovement: Float) extends TrainingListener {
  private var bestLoss = Float.MaxValue
  private var patienceCount = 0
  @volatile var shouldStop = false

  def getShouldStop: Boolean = shouldStop
  private var currentEpochLoss = 0f
  private var batchCount = 0

  override def onEpoch(trainer: Trainer): Unit = {
    if (batchCount > 0) {
      val avgLoss = currentEpochLoss / batchCount
      println(s"Average loss for epoch: $avgLoss (best: $bestLoss)")

      if (avgLoss < bestLoss * (1 - minImprovement)) {
        bestLoss = avgLoss
        patienceCount = 0
      } else {
        patienceCount += 1
        println(s"No improvement for $patienceCount epochs")
        if (patienceCount >= patience) {
          println(s"Early stopping triggered after $patienceCount epochs without improvement")
          shouldStop = true
        }
      }
    }
    // Reset for next epoch
    currentEpochLoss = 0f
    batchCount = 0
  }

  override def onTrainingBegin(trainer: Trainer): Unit = {
    if (trainer.getLoss != null ) {

    }
  }
  override def onTrainingEnd(trainer: Trainer): Unit = {
    println(s"Training ended with best loss: $bestLoss")
  }
  override def onTrainingBatch(trainer: Trainer, batch: BatchData): Unit = {
    if (shouldStop) {
      throw new RuntimeException("Early stopping triggered")
    }
  }
  override def onValidationBatch(trainer: Trainer, batch: BatchData): Unit = {

  }
}
