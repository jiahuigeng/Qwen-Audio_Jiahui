import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation import GenerationConfig
import torch
import argparse
import pickle
from typing import Tuple, List, Union, Iterable, Dict
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

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    audio_info = None
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            # import ipdb; ipdb.set_trace()
            audio_info = tokenizer.process_audio(content)
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.AUDIO_ST), audio_info=audio_info
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.AUDIO_ST), audio_info=audio_info),audio_info

        system_text, system_tokens_part, audio_info = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part, _ = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part, _ = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"


    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


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

    # input_ids = pickle.load(open("input_ids.bin", 'rb')).to(model.device)

    query = tokenizer.from_list_format([
        {'audio': 'assets/audio/1272-128104-0000.flac'},  # Either a local path or an url
        {'text': "translate the following sentence to English:"},
    ])

    raw_text, context_tokens = make_context(tokenizer, query)
    input_ids = torch.tensor([context_tokens]).to(device)
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
        context_length=len(context_tokens),
        chat_format=generation_config.chat_format,
        verbose=False,
        errors='replace',
        audio_info=audio_info
    )

    print("response", response)

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


