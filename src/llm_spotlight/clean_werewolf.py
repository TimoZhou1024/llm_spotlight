from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from datasets import VerificationMode, load_dataset
except ImportError:  # pragma: no cover
    VerificationMode = None
    load_dataset = None

from tqdm import tqdm

DEFAULT_DATASET = "ReneeYe/werewolf_game_reasoning"

PLAYER_NAMES = {
    1: "Alice",
    2: "Bob",
    3: "Charlie",
    4: "Diana",
    5: "Ethan",
    6: "Fiona",
    7: "George",
    8: "Hannah",
    9: "Ian",
    10: "Jack",
    11: "Kate",
    12: "Liam",
}

GAME_PROMPT = """You are playing a digital version of the social deduction game Werewolf (also known as Mafia).
- Keep your answers strictly under 500 words.

GAME RULES:
- Player Roles: players are divided into Werewolves, Villagers, and possible special roles such as Seer and Guard.
- Rounds consist of two phases:
    - Night Phase: Werewolves choose a player to eliminate. Special roles may investigate or protect players.
    - Day Phase: Players debate and vote to remove one player.
- Winning Conditions: Villagers win by voting out the Werewolves. Werewolves win when they can no longer be stopped by the Village."""

DEBATE_SCHEMA_HINT = """Follow the Werewolf Arena debate output style and produce exactly this JSON schema:

```json
{
  "reasoning": "string",
  "say": "string"
}
```"""

VOTE_SCHEMA_HINT = """Follow the Werewolf Arena vote output style and produce exactly this JSON schema:

```json
{
  "reasoning": "string",
  "vote": "string"
}
```"""

INVESTIGATE_SCHEMA_HINT = """Follow the Werewolf Arena investigate output style and produce exactly this JSON schema:

```json
{
  "reasoning": "string",
  "investigate": "string"
}
```"""

REMOVE_SCHEMA_HINT = """Follow the Werewolf Arena remove output style and produce exactly this JSON schema:

```json
{
  "reasoning": "string",
  "remove": "string"
}
```"""

PROTECT_SCHEMA_HINT = """Follow the Werewolf Arena protect output style and produce exactly this JSON schema:

```json
{
  "reasoning": "string",
  "protect": "string"
}
```"""

SUMMARIZE_SCHEMA_HINT = """Follow the Werewolf Arena summarize output style and produce exactly this JSON schema:

```json
{
    "reasoning": "string",
    "summary": "string"
}
```"""

PLAYER_TOKEN_RE = re.compile(
    r"\bPlayer\s*#?\s*(\d{1,2})(?=\b|and)|\b[Pp]layer\s*(\d{1,2})(?=\b|and)|\bNo\.\s*(\d{1,2})(?=\b|and)|\bNumber\s*(\d{1,2})(?=\b|and)",
    re.I,
)
PLAYER_LIST_RE = re.compile(r"Player\(s\)\s*((?:\d+\s*,\s*)*\d+)")
CJK_PUNCT_TRANSLATION = str.maketrans({"，": ", ", "。": ".", "；": "; ", "：": ": ", "！": "!", "？": "?"})

TRANSLATION_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bknife\s+(?:a\s+)?(?:person|people|player|players|someone)\b", re.I), "eliminate a player at night"),
    (re.compile(r"\bcut\s+(?:a\s+)?(?:person|people|player|players|someone)\b", re.I), "eliminate a player at night"),
    (re.compile(r"\b(?:eat|ate|eats|eating)\s+(?:a\s+)?knife\b", re.I), "was eliminated at night"),
    (re.compile(r"\b(?:was|were|is|am|be)\s+knifed\b", re.I), "was eliminated at night"),
    (re.compile(r"\bgo\s+to\s+police\b", re.I), "run for sheriff"),
    (re.compile(r"\bon\s+police\b", re.I), "running for sheriff"),
    (re.compile(r"\bjump\s+police\b", re.I), "run for sheriff"),
    (re.compile(r"\b(?:fierce|violent)\s+jump\b", re.I), "fake claim"),
    (re.compile(r"\bjump(?:ing)?\s+(?:as\s+)?(?:the\s+)?seer\b", re.I), "claim to be the Seer"),
    (re.compile(r"\bgold\s+water\b", re.I), "confirmed innocent"),
    (re.compile(r"\bsilver\s+water\b", re.I), "player saved by the Guard"),
    (re.compile(r"\bcheck\s+kills?\b", re.I), "Seer accusation"),
    (re.compile(r"\bkill\s+checks?\b", re.I), "Seer accusation"),
    (re.compile(r"\bbad\s+identity\b", re.I), "likely Werewolf-aligned player"),
    (re.compile(r"\bgood\s+identity\b", re.I), "likely Villager-aligned player"),
    (re.compile(r"\bresist\s+push\b", re.I), "be voted out"),
    (re.compile(r"\breturn\s+(?:ticket|vote)\b", re.I), "consolidate votes"),
    (re.compile(r"\bgather\s+(?:ticket|vote)s?\b", re.I), "consolidate votes"),
    (re.compile(r"\brush\s+(?:ticket|vote)s?\b", re.I), "vote together"),
    (re.compile(r"\breverse\s+hook\b", re.I), "bus a teammate"),
    (re.compile(r"\bback\s+hook\b", re.I), "bus a teammate"),
    (re.compile(r"\bseer\s*and\s+villager\b|\bseerand\s+villager\b", re.I), "Seer or Villager"),
    (re.compile(r"\bwerewolf\s*and\s+guard\b|\bwerewolfand\s+guard\b", re.I), "Werewolf or Guard"),
    (re.compile(r"\bvillager\s*and\s+werewolf\b|\bvillagerand\s+werewolf\b", re.I), "Villager or Werewolf"),
    (re.compile(r"\bwerewolf\s*and\s+villager\b|\bwerewolfand\s+villager\b", re.I), "Werewolf or Villager"),
]


@dataclass
class CleanStats:
    total: int = 0
    kept: int = 0
    dropped_parse_error: int = 0
    dropped_unsupported_schema: int = 0
    player_name_replacements: int = 0
    translation_fixes: int = 0


def normalize_whitespace(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(CJK_PUNCT_TRANSLATION)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def player_name(number: str | int) -> str:
    return PLAYER_NAMES.get(int(number), f"Player{int(number)}")


def replace_player_names(text: Any, stats: CleanStats) -> str:
    text = normalize_whitespace(text)

    def replace_list(match: re.Match[str]) -> str:
        names = [player_name(num) for num in re.findall(r"\d+", match.group(1))]
        stats.player_name_replacements += len(names)
        return ", ".join(names)

    text = PLAYER_LIST_RE.sub(replace_list, text)

    def replace_token(match: re.Match[str]) -> str:
        num = next(group for group in match.groups() if group is not None)
        stats.player_name_replacements += 1
        return player_name(num)

    return PLAYER_TOKEN_RE.sub(replace_token, text)


def fix_translated_terms(text: Any, stats: CleanStats) -> str:
    text = normalize_whitespace(text)
    for pattern, replacement in TRANSLATION_FIXES:
        text, count = pattern.subn(replacement, text)
        stats.translation_fixes += count
    return text


def clean_text(text: Any, stats: CleanStats) -> str:
    return fix_translated_terms(replace_player_names(text, stats), stats)


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def read_records(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input:
        path = Path(args.input)
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data
        return

    if load_dataset is None:
        raise RuntimeError("The datasets package is not installed. Run `uv sync` first.")
    kwargs: dict[str, Any] = {"split": args.split}
    if VerificationMode is not None:
        kwargs["verification_mode"] = VerificationMode.NO_CHECKS
    dataset = load_dataset(args.dataset, **kwargs)
    for row in dataset:
        yield dict(row)


def format_role_labels(labels: Any, stats: CleanStats) -> str:
    if not isinstance(labels, dict):
        return "unknown"
    parts = []
    for raw_name, raw_label in labels.items():
        name = clean_text(raw_name, stats)
        label = clean_text(raw_label, stats)
        parts.append(f"{name}: {label}")
    return "; ".join(parts) if parts else "unknown"


def build_speech_output(response: dict[str, Any], stats: CleanStats) -> dict[str, str] | None:
    speech = response.get("speech")
    if not speech:
        return None
    self_present = clean_text(response.get("self_present", "unknown"), stats)
    role_labels = format_role_labels(response.get("role_label"), stats)
    vote = clean_text(response.get("call_for_vote", "None"), stats)
    reasoning = (
        f"I plan to present myself as {self_present}. "
        f"My current read on other players is: {role_labels}. "
        f"My intended vote direction is {vote}."
    )
    return {"reasoning": clean_text(reasoning, stats), "say": clean_text(speech, stats)}


def build_vote_output(response: dict[str, Any], stats: CleanStats) -> dict[str, str] | None:
    vote = response.get("voting_player")
    reason = response.get("voting_reason") or response.get("notes")
    if vote in (None, "") or not reason:
        return build_identity_summary_output(response, stats)
    return {"reasoning": clean_text(reason, stats), "vote": clean_text(f"Player {vote}", stats)}


def build_identity_summary_output(response: dict[str, Any], stats: CleanStats) -> dict[str, str] | None:
    identity_items = []
    for key, value in response.items():
        if re.fullmatch(r"Player\s+\d{1,2}", str(key)):
            identity_items.append(f"{clean_text(key, stats)}: {clean_text(value, stats)}")
    if not identity_items:
        return None
    summary = "My current identity assessment is: " + "; ".join(identity_items) + "."
    reasoning = "I summarize each player's likely role based on the debate, night information, and voting context."
    return {"reasoning": reasoning, "summary": summary}


def build_action_output(response: dict[str, Any], stats: CleanStats) -> dict[str, str] | None:
    reason = clean_text(response.get("reason", ""), stats) or "I choose the most strategically useful target based on the current game state."
    if response.get("kill") not in (None, ""):
        return {"reasoning": reason, "remove": clean_text(f"Player {response['kill']}", stats)}
    if response.get("inquired") not in (None, ""):
        return {"reasoning": reason, "investigate": clean_text(f"Player {response['inquired']}", stats)}
    if response.get("guard") not in (None, ""):
        return {"reasoning": reason, "protect": clean_text(f"Player {response['guard']}", stats)}
    if response.get("heal") not in (None, "") and str(response.get("heal")) not in {"0", "None", "none", ""}:
        return {"reasoning": reason, "protect": clean_text(f"Player {response['heal']}", stats)}
    if response.get("poison") not in (None, "") and str(response.get("poison")) not in {"0", "None", "none", ""}:
        return {"reasoning": reason, "remove": clean_text(f"Player {response['poison']}", stats)}
    return None


def schema_hint(output: dict[str, str]) -> str:
    if "say" in output:
        return DEBATE_SCHEMA_HINT
    if "vote" in output:
        return VOTE_SCHEMA_HINT
    if "investigate" in output:
        return INVESTIGATE_SCHEMA_HINT
    if "protect" in output:
        return PROTECT_SCHEMA_HINT
    if "remove" in output:
        return REMOVE_SCHEMA_HINT
    if "summary" in output:
        return SUMMARIZE_SCHEMA_HINT
    raise ValueError("Unsupported output schema")


def build_instruction(row: dict[str, Any], output: dict[str, str], meta: dict[str, Any], stats: CleanStats) -> str:
    source_prompt = "\n\n".join(part for part in [row.get("instruction", ""), row.get("prompt", "")] if part)
    source_prompt = clean_text(source_prompt, stats)
    turn = clean_text(meta.get("turn", "unknown turn"), stats)
    role = clean_text(meta.get("role", "unknown role"), stats)
    name = player_name(meta.get("player_id", 0)) if meta.get("player_id") else "the current player"
    state = f"GAME STATE:\n- Current turn: {turn}.\n- You are {name} the {role}."
    return "\n\n".join([GAME_PROMPT, state, source_prompt, schema_hint(output)])


def convert_record(row: dict[str, Any], stats: CleanStats) -> dict[str, str] | None:
    try:
        meta = parse_json(row.get("meta"))
        response = parse_json(row.get("response"))
    except Exception:
        stats.dropped_parse_error += 1
        return None

    meta_type = str(meta.get("type", "")).lower()
    if meta_type == "speech":
        output = build_speech_output(response, stats)
    elif meta_type == "vote":
        output = build_vote_output(response, stats)
    elif meta_type == "action":
        output = build_action_output(response, stats)
    else:
        output = None

    if output is None:
        stats.dropped_unsupported_schema += 1
        return None

    instruction = build_instruction(row, output, meta, stats)
    return {
        "instruction": instruction,
        "input": "",
        "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
    }


def write_records(records: Iterable[dict[str, Any]], args: argparse.Namespace) -> CleanStats:
    stats = CleanStats()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(records, desc="cleaning"):
            stats.total += 1
            record = convert_record(row, stats)
            if record is None:
                if args.strict:
                    raise ValueError(f"Failed to convert row #{stats.total}")
                continue
            if args.dedupe:
                seen[record["output"]] += 1
                if seen[record["output"]] > 1:
                    continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.kept += 1
            if args.max_samples and stats.kept >= args.max_samples:
                break

    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean the English Werewolf reasoning dataset for LLaMA-Factory SFT.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train_en")
    parser.add_argument("--input", help="Optional local .json/.jsonl file for debugging.")
    parser.add_argument("--output", default="data/processed/werewolf_sft.jsonl")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dedupe", action="store_true", help="Drop exact duplicate output strings.")
    parser.add_argument("--strict", action="store_true", help="Raise if a row cannot be converted.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = write_records(read_records(args), args)
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
