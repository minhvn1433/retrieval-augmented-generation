class Retriever:
    def __init__(self, vectorstore, search_type, search_kwargs):
        self.vectorstore = vectorstore
        self.search_type = search_type
        self.search_kwargs = search_kwargs

    def get_relevant_documents(self, query):
        if self.search_type == "similarity":
            documents = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            documents = self.vectorstore.similarity_search_with_relevant_scores(
                query, **self.search_kwargs
            )

        return documents
