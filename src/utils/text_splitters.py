import copy

from utils.documents import Document


class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = start + self.chunk_size - self.chunk_overlap
        return chunks

    def create_documents(self, texts, metadatas):
        documents = []

        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadatas[i])
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)

        return documents

    def split_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return self.create_documents(texts, metadatas=metadatas)
