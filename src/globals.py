import torch
from sentence_transformers import CrossEncoder
from transformers import BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from utils.embeddings import SentenceTransformerEmbeddings
from utils.prompts import PromptTemplate
from utils.chat_models import ChatHuggingFace

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"batch_size": 16}
embedding = SentenceTransformerEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


retriever = None


prompt_template = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following \
pieces of retrieved context to answer the question. If you don't know \
the answer, just say that you don't know.

Context: 
{context}
    
Question: 
{query}"""
)
query_template = PromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five \
different versions of the given user question to retrieve relevant documents \
from a vector database. By generating multiple perspectives on the user \
question, your goal is to help the user overcome some of the limitations of \
the distance-based similarity search. Provide these alternative questions \
separated by newlines. Original question: {question}"""
)


bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
attn_implementation = (
    "flash_attention_2"
    if is_flash_attn_2_available() and torch.cuda.get_device_capability()[0] >= 8
    else "sdpa"
)
model_name = "google/gemma-2b-it"
tokenizer_kwargs = {}
model_kwargs = {
    "quantization_config": bnb_config,
    "low_cpu_mem_usage": False,
    "attn_implementation": attn_implementation,
}
generate_kwargs = {
    "temperature": 0.1,
    "do_sample": True,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
}
llm = ChatHuggingFace(
    model_name=model_name,
    tokenizer_kwargs=tokenizer_kwargs,
    model_kwargs=model_kwargs,
    generate_kwargs=generate_kwargs,
)


model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(model_name)


messages = []
