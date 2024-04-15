import re


class PromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def __str__(self):
        return f"input_variables={self.input_variables} template='{self.template}'"

    def format(self, **kwargs):
        return self.template.format(**kwargs)

    @classmethod
    def from_template(cls, template):
        input_variables = re.findall(r"{(\w+)}", template)
        return cls(template, input_variables)
