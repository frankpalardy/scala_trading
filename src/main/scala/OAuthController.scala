import org.springframework.stereotype.Controller
import org.springframework.web.bind.annotation.{GetMapping, RequestParam}
import org.springframework.web.servlet.view.RedirectView
import org.springframework.web.servlet.mvc.support.RedirectAttributes
import com.pangility.schwab.api.client.oauth2.SchwabAccount
import reactor.core.publisher.Mono

import scala.jdk.javaapi.FutureConverters
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

@Controller
class OAuthController(private val oauthManager: SchwabOAuthManager) {

  @GetMapping("/authorize")
  def initiateAuthorization(redirectAttributes: RedirectAttributes,
                            @RequestParam userId: String,
                            @RequestParam callback: String): RedirectView = {
    oauthManager.handleAuthorization(redirectAttributes, userId, callback)
  }

  @GetMapping("/refresh-with-account")
  def refreshTokenWithAccount(@RequestParam accountJson: String): RedirectView = {
    val account = parseAccountFromJson(accountJson)

    oauthManager.refreshAccessTokenWithAccount(account) match {
      case Some(monoAccount) =>
        monoAccount.subscribe(
          new java.util.function.Consumer[SchwabAccount] {
            def accept(refreshedAccount: SchwabAccount): Unit = {
              val refreshedAccountMono = Mono.just(refreshedAccount)
              handleMonoResult(refreshedAccountMono, "refresh-with-account")
            }
          },
          new java.util.function.Consumer[Throwable] {
            def accept(error: Throwable): Unit = {
              // Handle error
              println(s"Error refreshing account: ${error.getMessage}")
            }
          }
        )
        new RedirectView("/refresh-success")
      case None =>
        new RedirectView("/refresh-failed")
    }
  }

  // Helper method to parse JSON to SchwabAccount
  def parseAccountFromJson(accountJson: String): SchwabAccount = {
    // Implement JSON parsing logic
    new SchwabAccount()
  }

  @GetMapping("/refresh-with-userid")
  def refreshTokenWithUserId(@RequestParam userId: String): RedirectView = {
    oauthManager.refreshAccessTokenWithUserId(userId) match {
      case Some(monoAccount) =>
        handleMonoResult(monoAccount, "refresh-with-userid")
      case None =>
        new RedirectView("/refresh-failed")
    }
  }

  @GetMapping("/get-access-token")
  def getAccessToken(@RequestParam userId: String): RedirectView = {
    oauthManager.getAccessToken(userId) match {
      case Some(monoAccount) =>
        handleMonoResult(monoAccount, "get-access-token")
      case None =>
        new RedirectView("/get-token-failed")
    }
  }

  private def handleMonoResult(monoAccount: Mono[SchwabAccount], successPath: String): RedirectView = {
    val future = FutureConverters.asScala(monoAccount.toFuture())
    future.onComplete {
      case scala.util.Success(account) =>
        println(s"Operation successful for user: ${account.getUserId}")
      // Here you might want to store the account information or token
      case scala.util.Failure(error) =>
        println(s"Operation failed: ${error.getMessage}")
    }
    new RedirectView(s"/$successPath-result")
  }
}