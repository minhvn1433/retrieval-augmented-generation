from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(
        self, model_name=DEFAULT_MODEL_NAME, model_kwargs={}, encode_kwargs={}
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.embedding = SentenceTransformer(self.model_name, **self.model_kwargs)

    def __str__(self):
        return f"{self.embedding}"

    def embed_documents(self, texts):
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.embedding.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]
