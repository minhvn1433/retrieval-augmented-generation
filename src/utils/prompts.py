import re
from typing import List


class PromptTemplate:
    """Prompt template."""

    def __init__(self, template: str, input_variables: List[str]) -> None:
        """
        Create a new PromptTemplate.

        Args:
            template (str): Template string.
            input_variables (List[str]): List of input variables.
        """

        self.template = template
        self.input_variables = input_variables

    def __str__(self) -> str:
        """Return string representation of PromptTemplate."""

        return f"input_variables={self.input_variables} template='{self.template}'"

    def format(self, **kwargs) -> str:
        """Format template with input variables."""

        return self.template.format(**kwargs)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        """
        Create a new PromptTemplate.

        Args:
            template (str): Template string.

        Returns:
            PromptTemplate: Prompt template.
        """

        input_variables = re.findall(r"{(\w+)}", template)
        return cls(template, input_variables)
