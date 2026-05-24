import json
import os
from typing import Any, Dict, List

# Role names
VILLAGER = "Villager"
WEREWOLF = "Werewolf"
SEER = "Seer"
DOCTOR = "Doctor"


GAMEJSON = "game_complete.json"
ANAJSON = "processed_log.json"

from collections import Counter

def get_most_voted(vote):
    vote_counter = Counter(vote.values())
    if not vote_counter:
        return None
    max_votes = max(vote_counter.values())
    top_candidates = [name for name, count in vote_counter.items() if count == max_votes]

    if len(top_candidates) == 1:
        return top_candidates[0]
    else:
        return None

def reward(speaker:str,vote:dict,player_dict:dict):
    rewards = 0
    exiled = get_most_voted(vote)
    if exiled is None:
        return rewards
    
    if player_dict[exiled]['role'] == WEREWOLF:
        rewards = 3
    elif player_dict[exiled]['role'] == SEER or player_dict[exiled]['role'] == DOCTOR:
        rewards = -2
    else:
        rewards = -1

    if player_dict[speaker]['role'] == WEREWOLF:
        rewards = -rewards

    if exiled == speaker:
        rewards -= 0.5

    return rewards

def process(session_path):
    with open(os.path.join(session_path, GAMEJSON), "r") as f:
        data = json.load(f)

    player_dict = {}
    for entry in data["players"].values():
        name = entry["name"]
        role = entry["role"]
        player_dict[name] = {"role": role}

    new_log = {}
    for round in range(len(data["rounds"])-1):
        log_path = os.path.join(session_path, f"game_logs_{round}.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r") as f:
            logger = json.load(f)

        for names, entry in logger["logs"].items():
            if names == "round":
                continue
            if 'pvote' in entry and 'nvote' in entry and 'log' in entry:
                speaker = names.split("|")[0]

                p_rewards = reward(speaker, entry['pvote'], player_dict)
                n_rewards = reward(speaker, entry['nvote'], player_dict)

                # 构造唯一键（例如：f"{round_idx}_{speaker}"）
                # 或者保留原始 names 作为键，根据你的需求
                key = f"round{round}_{names}"

                new_log[key] = {
                    "log": entry["log"],
                    "reward_diff": p_rewards - n_rewards  # 或者分别存 p/n
                }

    return new_log

def get_winner(session_path):
    with open(os.path.join(session_path, GAMEJSON), "r") as f:
        data = json.load(f)
        
    if "winner" in data:
        return data["winner"]
    else:
        return None

def analysis(logs_root):
    counts = Counter()
    winners = Counter()

    for entry in os.listdir(logs_root):
        session_path = os.path.join(logs_root, entry)
        
        if os.path.isdir(session_path):
            print(f"\n正在处理会话目录: {entry}")

            complete_file = os.path.join(session_path, GAMEJSON)
            if os.path.exists(complete_file):
                new_log = process(session_path)

                # with open(os.path.join(session_path, "processed_log.json"), "w") as f:
                #     json.dump(new_log, f, ensure_ascii=False, indent=4)
                
                reward_diffs = [
                    entry["reward_diff"]
                    for entry in new_log.values()
                    if "reward_diff" in entry
                ]
                counts.update(reward_diffs)
                
                winner = get_winner(session_path)
                winners[winner] += 1


    # 打印结果s
    for value, count in sorted(counts.items()):
        print(f"reward_diff = {value}: {count} 次")

    for value,cout in winners.items():
        print(f"winner = {value}: {cout} 次")

def convert_kto(file_path: str, 
                            completion_field: str = "say",
                            include_metadata: bool = False) -> List[Dict[str, Any]]:
    """
    将狼人杀游戏数据文件转换为 KTO 训练格式
    
    Args:
        file_path: 输入JSON文件路径
        completion_field: 使用哪个字段作为completion ("say" 或 "full" 包含reasoning)
        include_metadata: 是否包含额外元数据字段
    
    Returns:
        KTO格式的数据列表，每条包含:
        - prompt: str, 输入提示
        - completion: str, 模型输出
        - label: bool, 偏好标签 (reward_diff > 0 为 True)
        - metadata: dict (可选), 额外信息
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
        KeyError: 缺少必需字段
    """
    kto_data = []
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 遍历每一轮数据
    for round_key, round_info in raw_data.items():
        try:
            # 第一层解析: log 字段是转义的JSON字符串
            log_data = json.loads(round_info['log'])
            
            # 第二层解析: raw_resp 字段也是转义的JSON字符串
            raw_resp = json.loads(log_data['raw_resp'])
            
            # 确定 completion 内容
            if completion_field == "full":
                completion = json.dumps(raw_resp, ensure_ascii=False)
            else:
                completion = raw_resp.get(completion_field, raw_resp.get('say', ''))
            
            # 确定偏好标签
            reward_diff = round_info.get('reward_diff', 0)
            if reward_diff > 4.0:
                label = True
            elif reward_diff <= -3.0:
                label = False
            else:
                continue
            
            # 构建KTO条目
            kto_entry = {
                "prompt": log_data['prompt'],
                "completion": completion,
                "label": label
            }
            
            # 可选：添加元数据
            if include_metadata:
                kto_entry['metadata'] = {
                    "round_id": round_key,
                    "reward_diff": reward_diff,
                    "reasoning": raw_resp.get('reasoning', '')
                }
            
            kto_data.append(kto_entry)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ 警告: 解析 {round_key} 时出错: {e}")
            continue
    
    return kto_data

def KTO():
    logs_root = "./logs"
    
    kto_data = []

    for entry in os.listdir(logs_root):
        session_path = os.path.join(logs_root, entry)
        
        if os.path.isdir(session_path):
            print(f"\n正在处理会话目录: {entry}")

            complete_file = os.path.join(session_path, GAMEJSON)
            convert_file = os.path.join(session_path, ANAJSON)
            if os.path.exists(complete_file) and os.path.exists(convert_file):
                new_log = convert_kto(convert_file, completion_field="full", include_metadata=False)
                kto_data.extend(new_log)
                print(f"  ✅ 已添加 {len(new_log)} 条数据，累计 {len(kto_data)} 条")
            else:
                print(f"  ⚠️ 文件不存在: {complete_file}")
    # 保存最终的KTO数据
    output_file = "kto_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kto_data, f, ensure_ascii=False, indent=2)
    print(f"\nKTO数据转换完成！共 {len(kto_data)} 条数据，已保存到 {output_file}")

if __name__ == "__main__":
    # KTO()
    analysis(logs_root = "./logs")