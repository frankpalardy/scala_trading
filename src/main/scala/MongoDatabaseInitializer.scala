import org.mongodb.scala.{MongoClient, MongoCollection}
import org.mongodb.scala.model.Indexes
import org.mongodb.scala.bson.collection.immutable.Document
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

object MongoConfig {
  private val uri = "mongodb://localhost:27017"
  private val databaseName = "trades"
  private val collectionName = "stocks"

  def getCollection: Future[MongoCollection[Document]] = {
    val client: MongoClient = MongoClient(uri)
    val database = client.getDatabase(databaseName)
    val collection = database.getCollection(collectionName)
    collection.createIndex(Indexes.compoundIndex(Indexes.ascending("symbol"), Indexes.ascending("date")))
      .toFuture()
      .map(_ => collection)
  }
}

object MongoDatabaseInitializer {
  def createTableAndLoadData(data: Seq[LongStockPrice]): Future[Unit] = {
    MongoConfig.getCollection.flatMap { collection =>
      insertData(collection, data)
    }.recover {
      case e: Exception =>
        println(s"Unexpected error: ${e.getMessage}")
    }
  }

  private def insertData(collection: MongoCollection[Document], data: Seq[LongStockPrice]): Future[Unit] = {
    val documents = data.map { stockPrice =>
      Document(
        "symbol" -> stockPrice.symbol,
        "date" -> stockPrice.date,
        "closePrice" -> stockPrice.closePrice,
        "timestamp" -> stockPrice.timestamp,
        "close" -> stockPrice.close      )
    }
    collection.insertMany(documents).toFuture().map(_ => ()).recover {
      case e: Exception => println(s"Error inserting documents: ${e.getMessage}")
    }
  }
}