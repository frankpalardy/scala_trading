object OptionPriceCalculator {

  def calculateAdjustedVolatility(
                                   stockDataWeek: List[LongStockPrice],
                                   vixPrice: Double,  // Current VIX price from Yahoo
                                   strikePrice: Double,
                                   timeToExpiration: Double,
                                   isCall: Boolean
                                 ): Double = {
    val currentPrice = stockDataWeek.maxBy(_.date).closePrice

    // Calculate base historical volatility
    val historicalVol = calculateHistoricalVolatility(stockDataWeek)

    // VIX is quoted in percentage points, convert to decimal
    val vixVol = vixPrice / 100.0

    // Use VIX as a scaling factor for our historical volatility
    val scalingFactor = vixVol / historicalVol
    val marketAdjustedVol = historicalVol * scalingFactor

    // Apply minimal skew based on moneyness
    val moneyness = strikePrice / currentPrice
    val skewAdjustment = if (isCall) {
      if (moneyness > 1.0) 1.05 else 0.95
    } else {
      if (moneyness < 1.0) 1.05 else 0.95
    }

    marketAdjustedVol * skewAdjustment
  }

  private def calculateHistoricalVolatility(stockDataWeek: List[LongStockPrice]): Double = {
    val dailyCloses = stockDataWeek.sortBy(_.date).map(_.closePrice)

    val returns = dailyCloses.sliding(2)
      .map { case Seq(p1, p2) => math.log(p2 / p1) }
      .filter(!_.isNaN)
      .toSeq

    if (returns.isEmpty) return 0.0

    val mean = returns.sum / returns.length
    val variance = returns.map(r => math.pow(r - mean, 2)).sum / returns.length

    math.sqrt(variance * 252)
  }


  def calculateOptionPrice(
                            currentPrice: Double,
                            strikePrice: Double,
                            timeToExpiration: Double,
                            riskFreeRate: Double,
                            volatility: Double,
                            isCall: Boolean
                          ): Double = {
    val d1 = (math.log(currentPrice/strikePrice) + (riskFreeRate + volatility*volatility/2)*timeToExpiration)/(volatility*math.sqrt(timeToExpiration))
    val d2 = d1 - volatility*math.sqrt(timeToExpiration)

    if (isCall)
      currentPrice*normalCDF(d1) - strikePrice*math.exp(-riskFreeRate*timeToExpiration)*normalCDF(d2)
    else
      strikePrice*math.exp(-riskFreeRate*timeToExpiration)*normalCDF(-d2) - currentPrice*normalCDF(-d1)
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
}
