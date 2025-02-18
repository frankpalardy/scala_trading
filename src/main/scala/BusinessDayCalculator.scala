object BusinessDayCalculator {
  // US Market Holidays (fixed and floating)
  private val fixedHolidays = Set(
    "01-01",  // New Year's Day
    "07-04",  // Independence Day
    "12-25"   // Christmas
  )

  private def isFloatingHoliday(date: java.util.Calendar): Boolean = {
    val month = date.get(java.util.Calendar.MONTH)
    val dayOfMonth = date.get(java.util.Calendar.DAY_OF_MONTH)
    val dayOfWeek = date.get(java.util.Calendar.DAY_OF_WEEK)
    val weekOfMonth = date.get(java.util.Calendar.WEEK_OF_MONTH)

    // Martin Luther King Jr. Day (3rd Monday in January)
    if (month == java.util.Calendar.JANUARY && dayOfWeek == java.util.Calendar.MONDAY && weekOfMonth == 3) return true

    // Presidents Day (3rd Monday in February)
    if (month == java.util.Calendar.FEBRUARY && dayOfWeek == java.util.Calendar.MONDAY && weekOfMonth == 3) return true

    // Memorial Day (Last Monday in May)
    if (month == java.util.Calendar.MAY && dayOfWeek == java.util.Calendar.MONDAY &&
      date.getActualMaximum(java.util.Calendar.DAY_OF_MONTH) - dayOfMonth < 7) return true

    // Labor Day (1st Monday in September)
    if (month == java.util.Calendar.SEPTEMBER && dayOfWeek == java.util.Calendar.MONDAY && weekOfMonth == 1) return true

    // Thanksgiving (4th Thursday in November)
    if (month == java.util.Calendar.NOVEMBER && dayOfWeek == java.util.Calendar.THURSDAY && weekOfMonth == 4) return true

    false
  }

  def calculateTimeToExpiration(expirationDate: String, useBusinessDays: Boolean = true): Double = {
    val dateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd")
    val targetDate = dateFormat.parse(expirationDate)
    val currentDate = new java.util.Date()

    if (useBusinessDays) {
      val calendar = java.util.Calendar.getInstance()
      var businessDays = 0
      calendar.setTime(currentDate)

      while (calendar.getTime.before(targetDate)) {
        // Check if it's a weekday
        val dayOfWeek = calendar.get(java.util.Calendar.DAY_OF_WEEK)
        if (dayOfWeek != java.util.Calendar.SATURDAY &&
          dayOfWeek != java.util.Calendar.SUNDAY) {

          // Check for holidays
          val monthDay = f"${calendar.get(java.util.Calendar.MONTH) + 1}%02d-${calendar.get(java.util.Calendar.DAY_OF_MONTH)}%02d"
          if (!fixedHolidays.contains(monthDay) && !isFloatingHoliday(calendar)) {
            businessDays += 1
          }
        }
        calendar.add(java.util.Calendar.DATE, 1)
      }

      businessDays.toDouble / 252.0
    } else {
      val daysInYear = 365.0
      (targetDate.getTime - currentDate.getTime) / (1000.0 * 60 * 60 * 24 * daysInYear)
    }
  }

  def isBusinessDay(date: java.util.Date): Boolean = {
    val calendar = java.util.Calendar.getInstance()
    calendar.setTime(date)

    val dayOfWeek = calendar.get(java.util.Calendar.DAY_OF_WEEK)
    if (dayOfWeek == java.util.Calendar.SATURDAY ||
      dayOfWeek == java.util.Calendar.SUNDAY) {
      return false
    }

    val monthDay = f"${calendar.get(java.util.Calendar.MONTH) + 1}%02d-${calendar.get(java.util.Calendar.DAY_OF_MONTH)}%02d"
    !fixedHolidays.contains(monthDay) && !isFloatingHoliday(calendar)
  }
}