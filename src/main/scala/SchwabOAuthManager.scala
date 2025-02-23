import com.pangility.schwab.api.client.oauth2.{SchwabAccount, SchwabOauth2Controller}
import org.springframework.ui.Model
import org.springframework.web.servlet.mvc.support.RedirectAttributes
import org.springframework.web.servlet.view.RedirectView
import reactor.core.publisher.Mono

import java.util
import java.util.function.{Consumer => JConsumer}
import scala.concurrent.{Future, Promise}


class SchwabOAuthManager {
  private val oauth2Controller = new SchwabOauth2Controller()


  def handleAuthorization(
                           redirectAttributes: RedirectAttributes,
                           userId: String,
                           callback: String
                         ): RedirectView = {
    oauth2Controller.authorize(redirectAttributes,userId, callback)
  }
  // Refresh Access Token with Account
  def refreshAccessTokenWithAccount(
                                     account: SchwabAccount
                                   ): Option[Mono[SchwabAccount]] = {
    try {
      val monoAccount = oauth2Controller.refreshAccessToken(account)
      Some(monoAccount)
    } catch {
      case e: Exception =>
        println(s"Token refresh with account failed: ${e.getMessage}")
        None
    }
  }

  // Refresh Access Token with UserID
  def refreshAccessTokenWithUserId(
                                    userId: String
                                  ): Option[Mono[SchwabAccount]] = {
    try {
      val monoAccount = oauth2Controller.refreshAccessToken(userId)
      Some(monoAccount)
    } catch {
      case e: Exception =>
        println(s"Token refresh with userId failed: ${e.getMessage}")
        None
    }
  }

  // Get Access Token with UserID
  def getAccessToken(
                      userId: String
                    ): Option[Mono[SchwabAccount]] = {
    try {
      val monoAccount = oauth2Controller.getAccessToken(userId)
      Some(monoAccount)
    } catch {
      case e: Exception =>
        println(s"Get access token failed: ${e.getMessage}")
        None
    }
  }
}

// Usage Example
object OAuthReactiveApp extends App {
  val oauthManager = new SchwabOAuthManager()
  val userId = "user123"
  val account = new SchwabAccount()

  // Example of handling Mono
  def handleTokenRefresh(): Unit = {
    // Refresh with Account
    oauthManager.refreshAccessTokenWithAccount(account).foreach { monoAccount =>
      monoAccount.subscribe(
        new JConsumer[SchwabAccount] {
          def accept(account: SchwabAccount): Unit = {
            println(s"Refreshed Account: ${account.getUserId}")
          }
        },
        new JConsumer[Throwable] {
          def accept(error: Throwable): Unit = {
            println(s"Error: ${error.getMessage}")
          }
        }
      )
    }

    // Refresh with UserID
    oauthManager.refreshAccessTokenWithUserId(userId).foreach { monoAccount =>
      monoAccount.subscribe(
        new JConsumer[SchwabAccount] {
          def accept(account: SchwabAccount): Unit = {
            println(s"Refreshed Account User ID: ${account.getUserId}")
          }
        },
        new JConsumer[Throwable] {
          def accept(error: Throwable): Unit = {
            println(s"Error refreshing account: ${error.getMessage}")
          }
        }
      )
    }
  }

  // You'll need to create RedirectAttributes
  val redirectAttributes = new RedirectAttributes() {
    override def addAttribute(attributeName: String, attributeValue: Any): RedirectAttributes = ???

    override def addAttribute(attributeValue: Any): RedirectAttributes = ???

    override def addAllAttributes(attributeValues: util.Collection[_]): RedirectAttributes = ???

    override def mergeAttributes(attributes: util.Map[String, _]): RedirectAttributes = ???

    override def addFlashAttribute(attributeName: String, attributeValue: Any): RedirectAttributes = ???

    override def addFlashAttribute(attributeValue: Any): RedirectAttributes = ???

    override def getFlashAttributes: util.Map[String, _] = ???

    override def addAllAttributes(attributes: util.Map[String, _]): Model = ???

    override def containsAttribute(attributeName: String): Boolean = ???

    override def getAttribute(attributeName: String): AnyRef = ???

    override def asMap(): util.Map[String, AnyRef] = ???
  }

  val redirectView = oauthManager.handleAuthorization(
    userId = "your_user_id",
    callback = "your_callback_url",
    redirectAttributes = redirectAttributes
  )
  // Run the example
  handleTokenRefresh()
}