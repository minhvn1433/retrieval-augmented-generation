class Document:
    """Document contains page content and metadata."""

    def __init__(self, page_content: str, metadata: dict | None = None) -> None:
        """
        Create a new Document.

        Args:
            page_content (str): String of page content.
            metadata (Optional[dict], optional): Dictionary of metadata. Defaults to None.
        """

        self.page_content = page_content
        self.metadata = metadata

    def __str__(self) -> str:
        """Return string representation of Document."""

        return f"page_content='{self.page_content}' metadata={self.metadata}"

    def __repr__(self) -> str:
        """Return string representation of Document."""

        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"
