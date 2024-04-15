class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __str__(self):
        return f"page_content='{self.page_content}' metadata={self.metadata}"

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"
