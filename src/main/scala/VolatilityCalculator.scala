object VolatilityCalculator {
  def calculateHistoricalVolatility(stockDataWeek: List[AssetPrice]): Double = {
    // Get daily closing prices from the week
    val dailyCloses = stockDataWeek.sortBy(_.date).map(_.closePrice)

    // Calculate daily returns
    val returns = dailyCloses.sliding(2)
      .map { case Seq(p1, p2) => math.log(p2 / p1) }
      .filter(!_.isNaN)
      .toSeq

    if (returns.isEmpty) return 0.0

    // Calculate standard deviation of returns
    val mean = returns.sum / returns.length
    val variance = returns.map(r => math.pow(r - mean, 2)).sum / returns.length

    // Annualize the volatility (252 trading days)
    math.sqrt(variance * 252)
  }

  def calculateParkinsonVolatility(stockDataWeek: List[AssetPrice]): Double = {
    // Calculate daily high-low volatility
    val dailyVolatilities = stockDataWeek.map { day =>
      val dayHigh = day.highs.max
      val dayLow = day.lows.min
      math.pow(math.log(dayHigh / dayLow), 2) / (4 * math.log(2))
    }.filter(!_.isNaN)

    if (dailyVolatilities.isEmpty) return 0.0

    // Annualize the volatility
    math.sqrt(252 * dailyVolatilities.sum / dailyVolatilities.length)
  }

  def calculateImpliedVolatility(
                                  stockDataWeek: List[AssetPrice],
                                  optionPrice: Double,
                                  strikePrice: Double,
                                  timeToExpiration: Double,
                                  riskFreeRate: Double,
                                  isCall: Boolean
                                ): Double = {
    // Get current price (latest close from most recent day)
    val currentPrice = stockDataWeek.maxBy(_.date).closePrice

    // Calculate historical and Parkinson volatilities
    val historicalVol = calculateHistoricalVolatility(stockDataWeek)
    val parkinsonVol = calculateParkinsonVolatility(stockDataWeek)

    // Use weighted average as initial estimate
    val initialVol = (historicalVol + 2 * parkinsonVol) / 3

    @annotation.tailrec
    def newtonRaphson(
                       vol: Double,
                       iterations: Int = 100,
                       tolerance: Double = 0.0001
                     ): Double = {
      if (iterations == 0) return vol

      val price = blackScholesPrice(currentPrice, strikePrice, timeToExpiration, riskFreeRate, vol, isCall)
      val vega = blackScholesVega(currentPrice, strikePrice, timeToExpiration, riskFreeRate, vol)

      val diff = price - optionPrice
      if (math.abs(diff) < tolerance) vol
      else {
        val newVol = vol - diff / vega
        if (newVol <= 0.0001 || newVol > 1000) vol
        else newtonRaphson(newVol, iterations - 1, tolerance)
      }
    }

    newtonRaphson(initialVol)
  }

  def calculateImpliedVolatilityWithSkew(
                                          stockDataWeek: List[AssetPrice],
                                          optionPrice: Double,
                                          strikePrice: Double,
                                          timeToExpiration: Double,
                                          riskFreeRate: Double,
                                          isCall: Boolean
                                        ): Double = {
    val currentPrice = stockDataWeek.maxBy(_.date).closePrice

    // Calculate moneyness (K/S ratio)
    val moneyness = strikePrice / currentPrice

    // Calculate base volatility
    val baseVol = calculateHistoricalVolatility(stockDataWeek)

    // Adjust initial volatility estimate based on moneyness
    val skewAdjustment = if (isCall) {
      if (moneyness > 1.0) {
        // OTM calls
        1.0 + 0.1 * math.pow(moneyness - 1.0, 2)
      } else {
        // ITM calls
        1.0 + 0.05 * math.pow(1.0 - moneyness, 2)
      }
    } else {
      if (moneyness < 1.0) {
        // OTM puts
        1.0 + 0.15 * math.pow(1.0 - moneyness, 2)
      } else {
        // ITM puts
        1.0 + 0.05 * math.pow(moneyness - 1.0, 2)
      }
    }

    val initialVol = baseVol * skewAdjustment

    @annotation.tailrec
    def newtonRaphson(
                       vol: Double,
                       iterations: Int = 100,
                       tolerance: Double = 0.0001
                     ): Double = {
      if (iterations == 0) return vol

      val price = blackScholesPrice(currentPrice, strikePrice, timeToExpiration, riskFreeRate, vol, isCall)
      val vega = blackScholesVega(currentPrice, strikePrice, timeToExpiration, riskFreeRate, vol)

      val diff = price - optionPrice
      if (math.abs(diff) < tolerance) vol
      else {
        val newVol = vol - diff / (vega + 0.0001) // Added small constant to prevent division by zero

        // Limit the adjustment in each iteration
        val maxAdjustment = 0.1
        val boundedNewVol = vol + math.max(-maxAdjustment, math.min(maxAdjustment, newVol - vol))

        // Apply reasonable bounds to volatility
        val finalNewVol = math.max(0.05, math.min(2.0, boundedNewVol))

        if (math.abs(finalNewVol - vol) < tolerance) vol
        else newtonRaphson(finalNewVol, iterations - 1, tolerance)
      }
    }

    // Add time-based adjustment for short-dated options
    val timeAdjustment = if (timeToExpiration < 0.1) {
      1.0 + (0.1 - timeToExpiration) * 0.5
    } else 1.0

    val finalVol = newtonRaphson(initialVol) * timeAdjustment

    // Ensure final result is within reasonable bounds
    math.max(0.05, math.min(2.0, finalVol))
  }

  private def blackScholesPrice(s: Double, k: Double, t: Double, r: Double, v: Double, isCall: Boolean): Double = {
    val d1 = (math.log(s/k) + (r + v*v/2)*t)/(v*math.sqrt(t))
    val d2 = d1 - v*math.sqrt(t)

    if (isCall)
      s*normalCDF(d1) - k*math.exp(-r*t)*normalCDF(d2)
    else
      k*math.exp(-r*t)*normalCDF(-d2) - s*normalCDF(-d1)
  }

  private def blackScholesVega(s: Double, k: Double, t: Double, r: Double, v: Double): Double = {
    val d1 = (math.log(s/k) + (r + v*v/2)*t)/(v*math.sqrt(t))
    s*math.sqrt(t)*normalPDF(d1)
  }

  private def normalCDF(x: Double): Double = {
    val a1 = 0.254829592
    val a2 = -0.284496736
    val a3 = 1.421413741
    val a4 = -1.453152027
    val a5 = 1.061405429
    val p = 0.3275911

    val sign = if (x < 0) -1 else 1
    val z = math.abs(x)/math.sqrt(2.0)

    val t = 1.0/(1.0 + p*z)
    val y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-z*z)

    0.5*(1.0 + sign*y)
  }

  private def normalPDF(x: Double): Double = {
    math.exp(-x*x/2) / math.sqrt(2*math.Pi)
  }
}

