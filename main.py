import json
import re
from vllm import LLM, SamplingParams

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

prompts = []
for task in tasks:
    prompt = system_prompt + f"Scenario:\n{task['scenario_context']}\nTarget Action Sequence:\n"
    prompts.append(prompt)

llm = LLM(model="./qwen_model_cache", dtype="auto", max_model_len=4096, trust_remote_code=True)

sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop=["Scenario:", "\n\n\n"])

outputs = llm.generate(prompts, sampling_params)

results = []
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text.strip()
    actions = re.findall(r"\(.*?\)", generated_text)

    results.append(
        {
            "assembly_task_id": tasks[i].get("assembly_task_id", f"task_{i}"),
            "complexity_level": len(actions),
            "target_action_sequence": actions,
        }
    )

with open("submission.json", "w") as f:
    json.dump(results, f, indent=4)
