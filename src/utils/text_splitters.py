import copy
import re
from typing import List

from utils.documents import Document


class CharacterTextSplitter:
    """Splitting text that looks at characters."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Create a new CharacterTextSplitter.

        Args:
            chunk_size (int, optional): Maximum size of chunks. Defaults to 1000.
            chunk_overlap (int, optional): Overlap in characters between chunks. Defaults to 200.

        Raises:
            ValueError: If `chunk_overlap` exceeds `chunk_size`.
        """

        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Chunk overlap size ({chunk_overlap}) is larger than chunk size ({chunk_size})."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into smaller chunks."""

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = start + self.chunk_size - self.chunk_overlap
        return chunks

    def create_documents(
        self, texts: List[str], metadatas: List[dict | None] | None = None
    ) -> List[Document]:
        """
        Create documents from texts.

        Args:
            texts (List[str]): List of texts.
            metadatas (List[dict | None] | None, optional): List of dictionaries of metadata. Defaults to None.

        Returns:
            List[Document]: List of documents.
        """

        documents = []

        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadatas[i]) if metadatas else None
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller document chunks.

        Args:
            documents (List[Document]): List of documents to split.

        Returns:
            List[Document]: List of splitted documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return self.create_documents(texts, metadatas=metadatas)


def split_text_with_regex(text: str, separator: str) -> List[str]:
    """Split text with a separator."""

    _splits = re.split(f"({separator})", text)

    splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
    if len(_splits) % 2 == 0:
        splits += _splits[-1:]
    splits = [_splits[0]] + splits

    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter:
    """Splitting text by recursively look at characters."""

    def __init__(
        self, chunk_size: int = 1000, separators: List[str] | None = None
    ) -> None:
        """
        Create a new RecursiveCharacterTextSplitter.

        Args:
            chunk_size (int, optional): Maximum size of chunks. Defaults to 1000.
            separators (List[str] | None, optional): List of separators. Defaults to None.
        """

        self.chunk_size = chunk_size
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits into chunks."""

        merged_texts = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
            else:
                merged_texts.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_texts.append(current_chunk)

        return merged_texts

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text into smaller chunks."""

        chunks = []

        # Get a separator
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if re.search(separator, text):
                separator = s
                new_separators = separators[i + 1 :]
                break

        splits = split_text_with_regex(text, separator)

        # Recursively splitting text
        good_splits = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_texts = self.merge_splits(good_splits)
                    chunks.extend(merged_texts)
                    good_splits = []
                if not new_separators:
                    chunks.append(s)
                else:
                    other_splits = self._split_text(s, new_separators)
                    chunks.extend(other_splits)

        if good_splits:
            merged_texts = self.merge_splits(good_splits)
            chunks.extend(merged_texts)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """Split text into smaller chunks."""

        return self._split_text(text, self.separators)

    def create_documents(
        self, texts: List[str], metadatas: List[dict | None] | None = None
    ) -> List[Document]:
        """
        Create documents from texts.

        Args:
            texts (List[str]): List of texts.
            metadatas (List[dict | None] | None, optional): List of dictionaries of metadata. Defaults to None.

        Returns:
            List[Document]: List of documents.
        """

        documents = []

        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadatas[i]) if metadatas else None
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller document chunks.

        Args:
            documents (List[Document]): List of documents to split.

        Returns:
            List[Document]: List of splitted documents.
        """

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return self.create_documents(texts, metadatas=metadatas)
