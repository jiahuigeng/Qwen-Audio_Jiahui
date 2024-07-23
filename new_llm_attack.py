# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model_name = "Qwen/Qwen-Audio-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True).eval()
#
# import torchaudio
#
# def load_audio(file_path):
#     waveform, sample_rate = torchaudio.load(file_path)
#     waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
#     return waveform.squeeze().numpy()
#
# audio_file = 'assets/audio/1089_134686_000007_000004.wav'
# audio_array = load_audio(audio_file)
#
# text_input = 'what does the person say?'
#
# # Tokenize text input
# text_inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
#
# # Prepare audio input (assuming audio_array is already in the correct format)
# audio_inputs = tokenizer(audio_array, return_tensors="pt", padding=True, truncation=True).to(model.device)
#
# # Combine inputs if necessary (this step may vary depending on the model's requirements)
# combined_inputs = {
#     'input_ids': text_inputs['input_ids'],
#     'audio_input_ids': audio_inputs['input_ids']  # Adjust according to model specifics
# }
#
# outputs = model.generate(**combined_inputs, max_new_tokens=100)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)