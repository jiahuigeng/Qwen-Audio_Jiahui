import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import argparse
import pickle
from qwen_generation_utils import decode_tokens
torch.manual_seed(1234)
audio_embed_file = f"audio_embed.bin"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()

trans_prompt = "translate the following sentence to English:"
tfd_audio_shape = (1, 80, 3000)

device = torch.device("cuda" if torch.cuda.is_available() else "")

# query = tokenizer.from_list_format([
#     {'audio': 'assets/audio/1272-128104-0000.flac'}, # Either a local path or an url
#     {'text': 'what does the person say?'},
# ])
#
# # response, history = model.chat(tokenizer, query=query, history=None)
# # print(response)
# raw_text = """'<|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Audio 1:<audio>assets/audio/1272-128104-0000.flac</audio>
# what does the person say?<|im_end|>
# <|im_start|>assistant
# '"""
#
# audio_info = tokenizer.process_audio(raw_text)



def main(args):

    target_prompt = None
    if args.task == "trans":
        target_prompt = trans_prompt

    # input_ids = tokenizer.encode(target_prompt)
    #
    # input_embeds = model.transformer.wte(torch.tensor(input_ids).to("cuda"))
    # tfd_audio_tensor = torch.randn(tfd_audio_shape).to("cuda").requires_grad_(True)

    raw_text = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Audio 1:<audio>assets/audio/1272-128104-0000.flac</audio>
translate the following sentence to English:<|im_end|>
<|im_start|>assistant
"""
    audio_shape = (1, 80, 300)
    input_audio_lengths = torch.tensor([[293, 146]])
    audio_span_tokens = [148]
    audios = torch.randn(audio_shape).to(device).requires_grad_(True)
    # tfd_audio_embedding = model.transformer.audio.encode(audios, input_audio_lengths, audio_span_tokens)

    input_audios = pickle.load(open("input_audios.bin", 'rb')).to(model.device)
    audio_info = {
        "input_audios": input_audios,
        "input_audio_lengths": input_audio_lengths,
        "audio_span_tokens": audio_span_tokens,
        'audio_urls': ['assets/audio/1272-128104-0001.flac']
    }
    kwargs = dict()
    kwargs['audio_info'] = audio_info
    stop_words_ids = [[151645], [151644]]
    generation_config = model.generation_config

    input_ids = pickle.load(open("input_ids.bin", 'rb')).to(model.device)

    outputs = model.generate(
        input_ids,
        stop_words_ids=stop_words_ids,
        return_dict_in_generate=False,
        generation_config=generation_config,
        **kwargs,
    )

    response = decode_tokens(
        outputs[0],
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=179,
        chat_format=generation_config.chat_format,
        verbose=False,
        errors='replace',
        audio_info=audio_info
    )

    print(response)

    # print(input_embeds)

# origin_audio_tensor = audio_info["input_audios"]
# pickle.dump(origin_audio_tensor, open(audio_embed_file, "wb"))





# audio_tensor = torch.randn(origin_audio_tensor.shape).to("cuda").requires_grad_(True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="trans")
    # parser.add_argument("")


    args = parser.parse_args()
    main(args)




# convert audio to mel spectrogram

# convert spectrogram to audio


