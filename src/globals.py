import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

# BI-ENCODER
bi_encoder_path = "sentence-transformers/all-MiniLM-L6-v2"
bi_encoder = SentenceTransformer(bi_encoder_path)

# CROSS-ENCODER
cross_encoder_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(cross_encoder_path)

# LLM
llm_path = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
attn_implementation = (
    "flash_attention_2"
    if is_flash_attn_2_available() and torch.cuda.get_device_capability()[0] >= 8
    else "sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(llm_path)
llm = AutoModelForCausalLM.from_pretrained(
    llm_path,
    quantization_config=bnb_config,
    attn_implementation=attn_implementation,
    low_cpu_mem_usage=True,
)

# GLOBAL
chat = []
