from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
import langchain_community.document_transformers
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

print(client)
print("##############")

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")

print(db)
print("######")
query = ("An alternative strategy that overcomes existing limitations of conventional approaches for exploring subcellular structure at nanoscale involves?")

docs = db.similarity_search_with_score(query=query, k=3)
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})


reordering = langchain_community.document_transformers.LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

for i in reordered_docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})