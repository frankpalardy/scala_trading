import scala.annotation.tailrec
import scala.math.{abs, exp, log, max, min, pow, sqrt}

object AmericanOptionGreeks {
  case class AmericanOptionGreeks(price: Double, impliedVolatility: Double, delta: Double, gamma: Double, theta: Double, vega: Double, rho: Double)

  def calculateAmericanOptionGreeks(
                                     underlyingPrice: Double,
                                     strikePrice: Double,
                                     timeToExpiration: Double, // in years
                                     riskFreeRate: Double,
                                     marketPrice: Double,
                                     isCall: Boolean,
                                     steps: Int = 100
                                   ): AmericanOptionGreeks = {
    val impliedVolatility = calculateImpliedVolatility(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, marketPrice, isCall, steps)

    val dt = timeToExpiration / steps
    val u = exp(impliedVolatility * sqrt(dt))
    val d = 1 / u
    val p = (exp(riskFreeRate * dt) - d) / (u - d)

    // Initialize the stock price tree
    val stockTree = Array.ofDim[Double](steps + 1, steps + 1)
    for (i <- 0 to steps) {
      for (j <- 0 to i) {
        stockTree(i)(j) = underlyingPrice * pow(u, j) * pow(d, i - j)
      }
    }

    // Initialize the option value tree
    val optionTree = Array.ofDim[Double](steps + 1, steps + 1)
    for (j <- 0 to steps) {
      optionTree(steps)(j) = math.max(0, if (isCall) stockTree(steps)(j) - strikePrice else strikePrice - stockTree(steps)(j))
    }

    // Backward induction
    for (i <- steps - 1 to 0 by -1) {
      for (j <- 0 to i) {
        val exerciseValue = math.max(0, if (isCall) stockTree(i)(j) - strikePrice else strikePrice - stockTree(i)(j))
        val continuationValue = exp(-riskFreeRate * dt) * (p * optionTree(i + 1)(j + 1) + (1 - p) * optionTree(i + 1)(j))
        optionTree(i)(j) = math.max(exerciseValue, continuationValue)
      }
    }

    // Calculate Greeks
    val price = optionTree(0)(0)
    val delta = (optionTree(1)(1) - optionTree(1)(0)) / (stockTree(1)(1) - stockTree(1)(0))
    val gamma = ((optionTree(2)(2) - optionTree(2)(1)) / (stockTree(2)(2) - stockTree(2)(1)) -
      (optionTree(2)(1) - optionTree(2)(0)) / (stockTree(2)(1) - stockTree(2)(0))) /
      ((stockTree(1)(1) - stockTree(1)(0)) / 2)
    val theta = (optionTree(1)(0) - price) / dt

    // For Vega and Rho, we need to recalculate with a small change in volatility and interest rate
    val dVol = 0.01
    val dRate = 0.01
    val priceWithHigherVol = calculateAmericanOptionPrice(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, impliedVolatility + dVol, isCall, steps)
    val priceWithHigherRate = calculateAmericanOptionPrice(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate + dRate, impliedVolatility, isCall, steps)

    val vega = (priceWithHigherVol - price) / dVol
    val rho = (priceWithHigherRate - price) / dRate

    AmericanOptionGreeks(price, impliedVolatility, delta, gamma, theta, vega, rho)
  }

  @tailrec
  private def calculateImpliedVolatility(
                                          underlyingPrice: Double,
                                          strikePrice: Double,
                                          timeToExpiration: Double,
                                          riskFreeRate: Double,
                                          marketPrice: Double,
                                          isCall: Boolean,
                                          steps: Int,
                                          volatility: Double = 0.5,
                                          tolerance: Double = 0.0001,
                                          maxIterations: Int = 100
                                        ): Double = {
    if (maxIterations <= 0) return volatility // Prevent infinite loops

    val price = calculateAmericanOptionPrice(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, volatility, isCall, steps)
    val diff = price - marketPrice

    if (abs(diff) < tolerance) {
      volatility
    } else {
      // Calculate vega
      val dVol = 0.001
      val priceUp = calculateAmericanOptionPrice(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, volatility + dVol, isCall, steps)
      val vega = (priceUp - price) / dVol

      // Newton-Raphson step
      val newVolatility = volatility - diff / (vega + 1e-10) // Add small value to prevent division by zero

      calculateImpliedVolatility(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, marketPrice, isCall, steps, newVolatility, tolerance, maxIterations - 1)
    }
  }

  def calculateAmericanOptionGreeksWithoutMarketPrice(
                                                       underlyingPrice: Double,
                                                       strikePrice: Double,
                                                       timeToExpiration: Double,
                                                       riskFreeRate: Double,
                                                       volatility: Double,
                                                       isCall: Boolean,
                                                       steps: Int = 100
                                                     ): AmericanOptionGreeks = {
    val dt = timeToExpiration / steps
    val u = exp(volatility * sqrt(dt))
    val d = 1 / u
    val p = (exp(riskFreeRate * dt) - d) / (u - d)

    // Initialize the stock price tree
    val stockTree = Array.ofDim[Double](steps + 1, steps + 1)
    for (i <- 0 to steps) {
      for (j <- 0 to i) {
        stockTree(i)(j) = underlyingPrice * pow(u, j) * pow(d, i - j)
      }
    }

    // Initialize the option value tree
    val optionTree = Array.ofDim[Double](steps + 1, steps + 1)
    for (j <- 0 to steps) {
      optionTree(steps)(j) = math.max(0, if (isCall) stockTree(steps)(j) - strikePrice else strikePrice - stockTree(steps)(j))
    }

    // Backward induction
    for (i <- steps - 1 to 0 by -1) {
      for (j <- 0 to i) {
        val exerciseValue = math.max(0, if (isCall) stockTree(i)(j) - strikePrice else strikePrice - stockTree(i)(j))
        val continuationValue = exp(-riskFreeRate * dt) * (p * optionTree(i + 1)(j + 1) + (1 - p) * optionTree(i + 1)(j))
        optionTree(i)(j) = math.max(exerciseValue, continuationValue)
      }
    }

    // Calculate Greeks
    val price = optionTree(0)(0)
    val delta = (optionTree(1)(1) - optionTree(1)(0)) / (stockTree(1)(1) - stockTree(1)(0))
    val gamma = ((optionTree(2)(2) - optionTree(2)(1)) / (stockTree(2)(2) - stockTree(2)(1)) -
      (optionTree(2)(1) - optionTree(2)(0)) / (stockTree(2)(1) - stockTree(2)(0))) /
      ((stockTree(1)(1) - stockTree(1)(0)) / 2)
    val theta = (optionTree(1)(0) - price) / dt

    // For Vega and Rho, we need to recalculate with a small change in volatility and interest rate
    val dVol = 0.01
    val dRate = 0.01
    val priceWithHigherVol = calculateAmericanOptionPrice(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate, volatility + dVol, isCall, steps)
    val priceWithHigherRate = calculateAmericanOptionPrice(underlyingPrice, strikePrice, timeToExpiration, riskFreeRate + dRate, volatility, isCall, steps)

    val vega = (priceWithHigherVol - price) / dVol
    val rho = (priceWithHigherRate - price) / dRate

    AmericanOptionGreeks(price, volatility, delta, gamma, theta, vega, rho)
  }

  private def calculateAmericanOptionPrice(
                                            underlyingPrice: Double,
                                            strikePrice: Double,
                                            timeToExpiration: Double,
                                            riskFreeRate: Double,
                                            volatility: Double,
                                            isCall: Boolean,
                                            steps: Int
                                          ): Double = {
    val dt = timeToExpiration / steps
    val u = exp(volatility * sqrt(dt))
    val d = 1 / u
    val p = (exp(riskFreeRate * dt) - d) / (u - d)

    val stockTree = Array.ofDim[Double](steps + 1, steps + 1)
    val optionTree = Array.ofDim[Double](steps + 1, steps + 1)

    for (i <- 0 to steps) {
      for (j <- 0 to i) {
        stockTree(i)(j) = underlyingPrice * pow(u, j) * pow(d, i - j)
      }
    }

    for (j <- 0 to steps) {
      optionTree(steps)(j) = math.max(0, if (isCall) stockTree(steps)(j) - strikePrice else strikePrice - stockTree(steps)(j))
    }

    for (i <- steps - 1 to 0 by -1) {
      for (j <- 0 to i) {
        val exerciseValue = math.max(0, if (isCall) stockTree(i)(j) - strikePrice else strikePrice - stockTree(i)(j))
        val continuationValue = exp(-riskFreeRate * dt) * (p * optionTree(i + 1)(j + 1) + (1 - p) * optionTree(i + 1)(j))
        optionTree(i)(j) = math.max(exerciseValue, continuationValue)
      }
    }

    optionTree(0)(0)
  }
}