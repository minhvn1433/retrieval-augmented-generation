class AIMessage:
    """Message from AI."""

    def __init__(self, content: str) -> None:
        """
        Create a new AIMessage.

        Args:
            content (str): AI message.
        """

        self.content = content

    def __str__(self) -> str:
        """Return string representation of AIMessage."""

        return f"content='{self.content}'"

    def __repr__(self) -> str:
        """Return string representation of AIMessage."""

        return f"AIMessage(content='{self.content}')"


class HumanMessage:
    """Message from human."""

    def __init__(self, content: str) -> None:
        """
        Create a new HumanMessage.

        Args:
            content (str): Human message.
        """

        self.content = content

    def __str__(self) -> str:
        """Return string representation of HumanMessage."""

        return f"content='{self.content}'"

    def __repr__(self) -> str:
        """Return string representation of HumanMessage."""

        return f"HumanMessage(content='{self.content}')"


class SystemMessage:
    """System message."""

    def __init__(self, content: str) -> None:
        """
        Create a new SystemMessage.

        Args:
            content (str): System message.
        """

        self.content = content

    def __str__(self) -> str:
        """Return string representation of SystemMessage."""
        return f"content='{self.content}'"

    def __repr__(self) -> str:
        """Return string representation of SystemMessage."""

        return f"SystemMessage(content='{self.content}')"
