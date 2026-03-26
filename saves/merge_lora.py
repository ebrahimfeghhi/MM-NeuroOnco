import argparse, torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration as AutoModelForVision2Seq
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="/home/ebrahim/MM-NeuroOnco/saves/neuroonco_gpt_nocot")
parser.add_argument("--out", default="/home/ebrahim/MM-NeuroOnco/saves/neuroonco_gpt_nocot_merged")
args = parser.parse_args()

base = "Qwen/Qwen3-VL-8B-Instruct"
adapter = args.adapter
out = args.out

print("Loading base model...")
model = AutoModelForVision2Seq.from_pretrained(base, torch_dtype=torch.bfloat16, trust_remote_code=True)
print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter)
print("Merging...")
model = model.merge_and_unload()
print("Saving...")
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoProcessor.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
print("Done!")
