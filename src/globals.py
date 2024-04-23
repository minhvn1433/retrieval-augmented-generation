import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

# BI-ENCODER
bi_encoder_path = r"C:\Users\nhatm\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\e4ce9877abf3edfe10b0d82785e83bdcb973e22e"
bi_encoder = SentenceTransformer(bi_encoder_path)

# CROSS-ENCODER
cross_encoder_path = r"C:\Users\nhatm\.cache\huggingface\hub\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\b2cfda50a1a9fc7919e7444afbb52610d268af92"
cross_encoder = CrossEncoder(cross_encoder_path)

# LLM
llm_path = r"C:\Users\nhatm\.cache\huggingface\hub\models--google--gemma-2b-it\snapshots\fdc848b27058c2d59aa1a4f563387e53b09fca97"
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
