from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_PATH = "./models/qwen/Qwen3-4B-Instruct-2507"
LORA_A = "./models/werewolf-qwen-lora2"
LORA_B = "./models/werewolf-kto-lora2"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype="auto"
)

# merge A
model = PeftModel.from_pretrained(base_model, LORA_A)
model = model.merge_and_unload()
model.save_pretrained("./models/sft/Qwen3-4B-SFT")

# merge B
model = PeftModel.from_pretrained(model, LORA_B)
model = model.merge_and_unload()

model.save_pretrained("./models/kto/Qwen3-4B-KTO")