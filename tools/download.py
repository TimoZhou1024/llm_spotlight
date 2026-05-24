from datasets import load_dataset
import json

dataset = load_dataset("ReneeYe/werewolf_game_reasoning", split="train_en")

def convert(example):
    return {
        "instruction": example["instruction"],
        "input": example["prompt"],
        "output": example["response"]
    }

new_ds = dataset.map(convert, remove_columns=dataset.column_names)

new_ds.to_json("werewolf_sft.json", orient="records", force_ascii=False)