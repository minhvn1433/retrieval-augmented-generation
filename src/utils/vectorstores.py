from sentence_transformers import util

from utils.retrievers import Retriever


class VectorStore:
    def __init__(self, documents, embeddings, embedding):
        self.documents = documents
        self.embeddings = embeddings
        self.embedding = embedding

    def similarity_search(self, query, k=4, **kwargs):
        query_embedding = self.embedding.embed_query(query)
        cos_sim_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        _, indices = cos_sim_scores.topk(k)
        documents = [self.documents[i] for i in indices]

        return documents

    def similarity_search_with_relevant_scores(self, query, k=4, **kwargs):
        query_embedding = self.embedding.embed_query(query)
        cos_sim_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        scores, indices = cos_sim_scores.topk(k)
        score_threshold = kwargs.get("score_threshold", 0.5)
        documents = [
            self.documents[i]
            for i, score in zip(indices, scores)
            if score > score_threshold
        ]

        return documents

    @classmethod
    def from_documents(cls, documents, embedding):
        texts = [doc.page_content for doc in documents]
        embeddings = embedding.embed_documents(texts)

        return cls(documents=documents, embeddings=embeddings, embedding=embedding)

    def as_retriever(self, search_type="similarity", search_kwargs={"k": 4}):
        return Retriever(
            vectorstore=self, search_type=search_type, search_kwargs=search_kwargs
        )
