import com.pangility.schwab.api.client.oauth2.{
  SchwabTokenHandler,
  SchwabOauth2Controller,
  RefreshTokenException
}
import java.time.Instant
import scala.concurrent.{Future, Promise}
import scala.util.{Try, Success, Failure}

class SchwabOAuthManager(
                          clientId: String,
                          clientSecret: String,
                          redirectUri: String
                        ) {
  // Create OAuth2 Controller
  private val oauth2Controller = new SchwabOauth2Controller(
    clientId,
    clientSecret,
    redirectUri
  )

  // Token Handler
  private val tokenHandler = new SchwabTokenHandler()

  // Generate Authorization URL
  def generateAuthorizationUrl(): String = {
    oauth2Controller.getAuthorizationUrl()
  }

  // Handle Authorization
  def handleAuthorization(authorizationCode: String): Future[String] = {
    val promise = Promise[String]()

    Try {
      // Exchange authorization code for access token
      val accessToken = oauth2Controller.exchangeAuthorizationCode(authorizationCode)
      promise.success(accessToken)
    } match {
      case Success(_) =>
        promise.future
      case Failure(ex) =>
        promise.failure(new Exception(s"Authorization failed: ${ex.getMessage}"))
        promise.future
    }

    promise.future
  }

  // Refresh Token
  def refreshAccessToken(refreshToken: String): Future[String] = {
    val promise = Promise[String]()

    Try {
      val newAccessToken = tokenHandler.refreshAccessToken(refreshToken)
      promise.success(newAccessToken)
    } match {
      case Success(_) =>
        promise.future
      case Failure(ex: RefreshTokenException) =>
        promise.failure(new Exception(s"Token refresh failed: ${ex.getMessage}"))
        promise.future
      case Failure(ex) =>
        promise.failure(new Exception(s"Unexpected error: ${ex.getMessage}"))
        promise.future
    }

    promise.future
  }

  // Validate Token
  def isTokenValid(token: String): Boolean = {
    try {
      tokenHandler.validateAccessToken(token)
      true
    } catch {
      case _: Exception => false
    }
  }

  // Revoke Token
  def revokeToken(token: String): Future[Boolean] = {
    val promise = Promise[Boolean]()

    Try {
      oauth2Controller.revokeToken(token)
      promise.success(true)
    } match {
      case Success(_) =>
        promise.future
      case Failure(ex) =>
        promise.failure(new Exception(s"Token revocation failed: ${ex.getMessage}"))
        promise.future
    }

    promise.future
  }
}

// Usage Example
object SchwabOAuthApp extends App {
  val oauthManager = new SchwabOAuthManager(
    clientId = sys.env.getOrElse("SCHWAB_CLIENT_ID", ""),
    clientSecret = sys.env.getOrElse("SCHWAB_CLIENT_SECRET", ""),
    redirectUri = sys.env.getOrElse("SCHWAB_REDIRECT_URI", "")
  )

  // OAuth Flow Example
  def performOAuthFlow(): Unit = {
    // Generate Authorization URL
    val authUrl = oauthManager.generateAuthorizationUrl()
    println(s"Please visit this URL and authorize: $authUrl")

    // After getting authorization code
    val authorizationCode = "your_authorization_code"

    val authFuture = oauthManager.handleAuthorization(authorizationCode)

    authFuture.foreach { accessToken =>
      println(s"Access Token: $accessToken")

      // Validate token
      if (oauthManager.isTokenValid(accessToken)) {
        println("Token is valid")
      }
    }
  }
}

// Akka Actor Integration
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors

object OAuthActor {
  sealed trait OAuthCommand
  case class Authorize(code: String) extends OAuthCommand
  case class RefreshToken(token: String) extends OAuthCommand

  def apply(oauthManager: SchwabOAuthManager): Behavior[OAuthCommand] =
    Behaviors.receiveMessage {
      case Authorize(code) =>
        oauthManager.handleAuthorization(code)
        Behaviors.same
      case RefreshToken(token) =>
        oauthManager.refreshAccessToken(token)
        Behaviors.same
    }
}