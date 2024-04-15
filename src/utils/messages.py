class AIMessage:
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return f"content='{self.content}'"

    def __repr__(self):
        return f"AIMessage(content='{self.content}')"


class HumanMessage:
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return f"content='{self.content}'"

    def __repr__(self):
        return f"HumanMessage(content='{self.content}')"


class SystemMessage:
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return f"content='{self.content}'"

    def __repr__(self):
        return f"SystemMessage(content='{self.content}')"
