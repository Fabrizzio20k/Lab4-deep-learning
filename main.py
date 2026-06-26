import json
import re
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with open("Examples.json", "r") as f:
    examples = json.load(f)

with open("Task.json", "r") as f:
    tasks = json.load(f)

system_prompt = "You are an expert logical planner. You solve the scenarios by providing the exact target action sequence required to achieve the goal.\n\n"

for ex in examples:
    system_prompt += (
        f"Scenario:\n{ex['scenario_context']}\nTarget Action Sequence:\n"
        + "\n".join(ex["target_action_sequence"])
        + "\n\n"
    )

prompts = [
    system_prompt + f"Scenario:\n{task['scenario_context']}\nTarget Action Sequence:\n"
    for task in tasks
]

# APUNTAMOS A LA RUTA LOCAL DESCARGADA
MODEL_PATH = "./qwen_model_cache"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=quantization_config,
)

BATCH_SIZE = 4
results = []

for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    batch_tasks = tasks[i : i + BATCH_SIZE]

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_length:]
    decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for j, output_text in enumerate(decoded_outputs):
        actions = re.findall(r"\(.*?\)", output_text.strip())
        results.append(
            {
                "assembly_task_id": batch_tasks[j].get("assembly_task_id", f"task_{i+j}"),
                "complexity_level": len(actions),
                "target_action_sequence": actions,
            }
        )

    del inputs, outputs, generated_tokens
    gc.collect()
    torch.cuda.empty_cache()

with open("submission.json", "w") as f:
    json.dump(results, f, indent=4)
