

import scala.annotation.tailrec
import scala.math.{abs, exp, log, max, min, pow, sqrt}
import scala.math._

object OptionGreeks {
  case class OptionGreeks(impliedVolatility: Double, delta: Double, gamma: Double, theta: Double, vega: Double, rho: Double)

  def calculateOptionGreeks(
                             marketPrice: Double,
                             underlyingPrice: Double,
                             strikePrice: Double,
                             timeToExpiration: Double, // in years
                             riskFreeRate: Double,
                             isCall: Boolean
                           ): OptionGreeks = {
    // First, calculate implied volatility using Newton-Raphson method
    val impliedVolatility = calculateImpliedVolatility(marketPrice, underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, isCall)

    // Calculate d1 and d2
    val d1 = (log(underlyingPrice / strikePrice) + (riskFreeRate + pow(impliedVolatility, 2) / 2) * timeToExpiration) / (impliedVolatility * sqrt(timeToExpiration))
    val d2 = d1 - impliedVolatility * sqrt(timeToExpiration)

    // Calculate Greeks
    val delta = if (isCall) cumulativeNormalDistribution(d1) else -cumulativeNormalDistribution(-d1)
    val gamma = normalPDF(d1) / (underlyingPrice * impliedVolatility * sqrt(timeToExpiration))
    val vega = underlyingPrice * sqrt(timeToExpiration) * normalPDF(d1) / 100 // Divided by 100 to express in terms of 1% change in volatility
    val theta = if (isCall) {
      (-underlyingPrice * normalPDF(d1) * impliedVolatility / (2 * sqrt(timeToExpiration))
        - riskFreeRate * strikePrice * exp(-riskFreeRate * timeToExpiration) * cumulativeNormalDistribution(d2)) / 365
    } else {
      (-underlyingPrice * normalPDF(d1) * impliedVolatility / (2 * sqrt(timeToExpiration))
        + riskFreeRate * strikePrice * exp(-riskFreeRate * timeToExpiration) * cumulativeNormalDistribution(-d2)) / 365
    }
    val rho = if (isCall) {
      strikePrice * timeToExpiration * exp(-riskFreeRate * timeToExpiration) * cumulativeNormalDistribution(d2) / 100
    } else {
      -strikePrice * timeToExpiration * exp(-riskFreeRate * timeToExpiration) * cumulativeNormalDistribution(-d2) / 100
    }

    OptionGreeks(impliedVolatility, delta, gamma, theta, vega, rho)
  }

  @tailrec
  private def calculateImpliedVolatility(
                                          marketPrice: Double,
                                          underlyingPrice: Double,
                                          strikePrice: Double,
                                          timeToExpiration: Double,
                                          riskFreeRate: Double,
                                          isCall: Boolean,
                                          volatility: Double = 0.2,
                                          tolerance: Double = 0.00001,
                                          maxIterations: Int = 100
                                        ): Double = {
    if (maxIterations <= 0) return volatility // Prevent infinite loops

    val price = blackScholesPrice(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, volatility, isCall)
    val vega = blackScholesVega(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, volatility)

    val diff = marketPrice - price
    if (abs(diff) < tolerance) {
      volatility
    } else {
      val adjustmentFactor = 0.1
      val newVolatility = volatility + (diff / (vega + 0.0001)) * adjustmentFactor

      // Allow for a much wider range of volatility values
      val clampedVolatility = max(0.00001, min(newVolatility, 2))

      calculateImpliedVolatility(marketPrice, underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, isCall, clampedVolatility, tolerance, maxIterations - 1)
    }

  }

  private def blackScholesPrice(
                                 underlyingPrice: Double,
                                 strikePrice: Double,
                                 timeToExpiration: Double,
                                 riskFreeRate: Double,
                                 volatility: Double,
                                 isCall: Boolean
                               ): Double = {
    val (d1, d2) = calculateD1AndD2(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, volatility)

    if (isCall) {
      underlyingPrice * cumulativeNormalDistribution(d1) - strikePrice * exp(-riskFreeRate * timeToExpiration) * cumulativeNormalDistribution(d2)
    } else {
      strikePrice * exp(-riskFreeRate * timeToExpiration) * cumulativeNormalDistribution(-d2) - underlyingPrice * cumulativeNormalDistribution(-d1)
    }
  }

  private def blackScholesVega(
                                underlyingPrice: Double,
                                strikePrice: Double,
                                timeToExpiration: Double,
                                riskFreeRate: Double,
                                volatility: Double
                              ): Double = {
    val (d1, _) = calculateD1AndD2(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, volatility)
    underlyingPrice * sqrt(timeToExpiration) * normalPDF(d1)
  }

  private def normalPDF(x: Double): Double = {
    exp(-pow(x, 2) / 2) / sqrt(2 * Pi)
  }

  // Helper function to calculate the cumulative normal distribution
  private def cumulativeNormalDistribution(x: Double): Double = {
    val t = 1.0 / (1.0 + 0.2316419 * abs(x))
    val d = 0.3989423 * exp(-pow(x, 2) / 2.0)
    val probability = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    if (x > 0) 1.0 - probability else probability
  }

  private def calculateD1AndD2(
                                underlyingPrice: Double,
                                strikePrice: Double,
                                timeToExpiration: Double,
                                riskFreeRate: Double,
                                volatility: Double
                              ): (Double, Double) = {
    val adjustedTime = max(timeToExpiration, 0.0001) // Prevent division by zero
    val adjustedVolatility = max(volatility, 0.0001) // Prevent division by zero

    val d1 = (log(underlyingPrice / strikePrice) + (riskFreeRate + pow(adjustedVolatility, 2) / 2) * adjustedTime) / (adjustedVolatility * sqrt(adjustedTime))
    val d2 = d1 - adjustedVolatility * sqrt(adjustedTime)

    (d1, d2)
  }
}