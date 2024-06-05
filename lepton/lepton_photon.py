import sys
from leptonai.photon import Photon
import torch
import warnings
import os
import time
# disable some warnings
warnings.filterwarnings("ignore")
# reference: https://github.com/leptonai/examples/blob/main/getting-started/extra_files/main.py


class Main(Photon):
    requirement_dependency = ["flash-attn"]


    def init(self):
        model_name = "NexaAIDev/Octopus-v2"
        sys.path.append(os.getcwd())
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_default_device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.template = "Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {prompt} \n\nResponse:"


    def inference_helper(self, model, tokenizer, input_ids, eos_token_id, temperature):
        input_length = input_ids["input_ids"].shape[1]
        start_time = time.time()
        if eos_token_id >= 0:
            outputs = model.generate(
            input_ids=input_ids["input_ids"], 
                max_length= len(input_ids["input_ids"][0]) + 1024,
                temperature=temperature,
                do_sample=False,
                eos_token_id=eos_token_id)
        # https://huggingface.co/nexaai-unreleased/octopus-v4-streaming/blob/main/tokenizer_config.json#L357-L365
        else:
            outputs = model.generate(
                input_ids=input_ids["input_ids"], 
                max_length= len(input_ids["input_ids"][0]) + 1024,
                temperature=temperature,
                do_sample=False)
        end_time = time.time()
        generated_sequence = outputs[:, input_length:].tolist()
        res = tokenizer.decode(generated_sequence[0])
        return {"output": res, "latency": end_time - start_time}


    @Photon.handler("inference")
    def inference(self, prompt: str, eos_token_id: int, temperature: float) -> str:
        text = self.template.format(prompt = prompt)
        input_ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        return self.inference_helper(self.model,self.tokenizer,input_ids,eos_token_id,temperature)


if __name__ == "__main__":
	print(torch.cuda.is_available())

    # nexa_photon = Main()
    # eos_token_id = 256021
    # temperature = 0
    # prompt = "I want to take a photo"
    # print(nexa_photon.inference(prompt=prompt, eos_token_id=eos_token_id, temperature=temperature))