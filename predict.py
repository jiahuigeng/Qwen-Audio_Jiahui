from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()

query = tokenizer.from_list_format([
    # {'audio': 'assets/audio/1272-128104-0000.flac'},  # Either a local path or an url
    {'text': "translate the following sentence to English:"},
])


response, history = model.chat(tokenizer, query=query, history=None)
print(response)                                                                                                                                              