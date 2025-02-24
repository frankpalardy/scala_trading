import org.springframework.context.annotation.{Bean, Configuration}
import org.springframework.beans.factory.annotation.Value

@Configuration
class OAuthConfig {
  @Value("${schwab.client.id}")
  private val clientId: String = ""

  @Value("${schwab.client.secret}")
  private val clientSecret: String = ""

  @Value("${schwab.redirect.uri}")
  private val redirectUri: String = ""

  @Bean
  def schwabOAuthManager(): SchwabOAuthManager = {
    new SchwabOAuthManager()
  }
}