import torch

model_path = "/model-weights/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",    
    attn_implementation="flash_attention_2"
)
