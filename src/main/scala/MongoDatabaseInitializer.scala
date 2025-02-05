import org.mongodb.scala.{MongoClient, MongoCollection, Document}
import org.mongodb.scala.model.Indexes
import scala.concurrent.ExecutionContext.Implicits.global

object MongoConfig {
  val uri = "mongodb://localhost:27017"
  val databaseName = "trades"
  val collectionName = "stocks"

  def getCollection: MongoCollection[Document] = {
    val client: MongoClient = MongoClient(uri)
    val database = client.getDatabase(databaseName)
    val collection = database.getCollection(collectionName)
    collection.createIndex(Indexes.compoundIndex(Indexes.ascending("symbol"), Indexes.ascending("date"))).toFuture()
    collection
  }
}

object MongoDatabaseInitializer {
  def createTableAndLoadData(data: Seq[StockPrice]): Unit = {
    val collection = MongoConfig.getCollection
    try {
      insertData(collection, data)
    } catch {
      case e: Exception =>
        println(s"Unexpected error: ${e.getMessage}")
    }
  }

  private def insertData(collection: MongoCollection[Document], data: Seq[StockPrice]): Unit = {
    val documents = data.map { stockPrice =>
      Document(
        "symbol" -> stockPrice.symbol,
        "date" -> stockPrice.date,
        "closePrice" -> stockPrice.closePrice
      )
    }
    collection.insertMany(documents).toFuture().recover {
      case e: Exception => println(s"Error inserting documents: ${e.getMessage}")
    }
  }
}