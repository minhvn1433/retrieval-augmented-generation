from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.messages import AIMessage, HumanMessage, SystemMessage


class ChatHuggingFace:
    DEFAULT_MODEL_NAME = "google/gemma-2b-it"

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        tokenizer_kwargs={},
        model_kwargs={},
        generate_kwargs={},
    ):
        self.model_name = model_name
        self.tokenizer_kwargs = tokenizer_kwargs
        self.model_kwargs = model_kwargs
        self.generate_kwargs = generate_kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **self.tokenizer_kwargs
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name, **self.model_kwargs
        )

    def __str__(self):
        return f"{self.llm}"

    def generate(self, messages):
        messages_dicts = [self.to_chatml_format(m) for m in messages]
        input_text = self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.llm.device)
        outputs = self.llm.generate(**inputs, **self.generate_kwargs)
        output_text = self.tokenizer.decode(outputs[0])
        response = AIMessage(
            content=output_text.replace(input_text, "")
            .replace("<bos>", "")
            .replace("<eos>", "")
        )

        return response

    def to_chatml_format(self, message):
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}
