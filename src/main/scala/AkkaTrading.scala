import akka.actor.{Actor, ActorRef, ActorSystem, Props}

import scala.collection.mutable

object AkkaTrading {
  // Messages
  case class Buy(asset: Asset, quantity: Int)
  case class Sell(asset: Asset, quantity: Int)
  case object GetPortfolio
  case object GetTotalValue
  case class UpdatePrice(symbol: String, newPrice: Double)
  case object GetHighestValueAsset
  case object GetLowestValueAsset

  // Responses
  case class PortfolioResponse(portfolio: Map[String, (Asset, Int)])
  case class TotalValueResponse(value: Double)
  case class AssetResponse(asset: Option[(Asset, Int)])

  def props: Props = Props(new AkkaTradingActor)
}

class AkkaTradingActor extends Actor {
  import AkkaTrading._

  private val assets: mutable.Map[String, (Asset, Int)] = mutable.Map()

  def receive: Receive = {
    case Buy(asset, quantity) =>
      assets.updateWith(asset.symbol) {
        case Some((existingAsset, existingQuantity)) => Some((existingAsset, existingQuantity + quantity))
        case None => Some((asset, quantity))
      }

    case Sell(asset, quantity) =>
      assets.updateWith(asset.symbol) {
        case Some((existingAsset, existingQuantity)) if existingQuantity >= quantity =>
          if (existingQuantity == quantity) None else Some((existingAsset, existingQuantity - quantity))
        case other => other
      }

    case GetPortfolio =>
      sender() ! PortfolioResponse(assets.toMap)

    case GetTotalValue =>
      val totalValue = assets.values.map { case (asset, quantity) => asset.closePrice * quantity }.sum
      sender() ! TotalValueResponse(totalValue)

    case UpdatePrice(symbol, newPrice) =>
      assets.updateWith(symbol) {
        case Some((asset, quantity)) => Some((asset.copy(closePrice = newPrice), quantity))
        case None => None
      }

    case GetHighestValueAsset =>
      val highestValueAsset = assets.values.maxByOption { case (asset, quantity) => asset.closePrice * quantity }
      sender() ! AssetResponse(highestValueAsset)

    case GetLowestValueAsset =>
      val lowestValueAsset = assets.values.minByOption { case (asset, quantity) => asset.closePrice * quantity }
      sender() ! AssetResponse(lowestValueAsset)
  }
}

// Usage example
class AkkaTradingExample extends App {
  private val system = ActorSystem("TradingSystem")
  private val tradingActor: ActorRef = system.actorOf(AkkaTrading.props, "tradingActor")

  // Example usage
  tradingActor ! AkkaTrading.Buy(Asset("AAPL", "2023-05-20", 150.0, Seq.empty, Seq.empty, Seq.empty, Seq.empty, Seq.empty), 10)

  // To get the portfolio, we need to use the ask pattern
  import akka.pattern.ask
  import akka.util.Timeout
  import scala.concurrent.duration._
  import scala.concurrent.ExecutionContext.Implicits.global

  implicit val timeout: Timeout = Timeout(5.seconds)

  (tradingActor ? AkkaTrading.GetPortfolio).mapTo[AkkaTrading.PortfolioResponse].foreach { response =>
    println(s"Portfolio: ${response.portfolio}")
  }

  // Shutdown the ActorSystem after a delay
  system.scheduler.scheduleOnce(6.seconds) {
    system.terminate()
  }
}