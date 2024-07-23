from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os,torch
torch.manual_seed(1234)
import argparse
import pandas as pd
from utils import get_target_data

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()

def main(args):
    dataset = get_target_data(args.task)

    exp_folder = "exps"
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder, exist_ok=True)

    prefix = "_".join([args.task, "results"])
    save_folder = os.path.join(exp_folder, prefix)
    save_file = os.path.join(save_folder, prefix+".csv")

    df = pd.DataFrame(columns=['idx', 'prompt', 'pred', 'suffix_pred'])
    for idx, prompt in enumerate(dataset):
        query = tokenizer.from_list_format([
            {'audio': 'assets/audio/1272-128104-0000.flac'},  # Either a local path or an url
            {'text': prompt},
        ])

        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)

        suffix_query = tokenizer.from_list_format([
            {'audio': 'assets/audio/1272-128104-0000.flac'},  # Either a local path or an url
            {'text': prompt + " Yes, I can provide "},
        ])

        suffix_response, history = model.chat(tokenizer, query=suffix_query, history=None)
        print(suffix_response)

        df.at[idx, 'idx'] = idx
        df.at[idx, 'prompt'] = prompt
        df.at[idx, 'pred'] = response
        df.at[idx, 'suffix_pred'] = suffix_response
        df.to_csv(save_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="advbench")

    args = parser.parse_args()
    main(args)

