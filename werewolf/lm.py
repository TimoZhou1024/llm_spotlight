# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Any, Dict, Iterator, List, Optional

import json
import jinja2
from werewolf import utils
from werewolf.utils import Deserializable
from werewolf import apis
from werewolf.config import RETRIES


@dataclasses.dataclass
class LmLog(Deserializable):
    prompt: str
    raw_resp: str
    result: Any

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        return cls(**data)
    
    def to_json_minimal(self) -> str:
        return json.dumps({
            "prompt": self.prompt,
            "raw_resp": self.raw_resp
        })


@dataclasses.dataclass
class StreamingGeneration:
    prompt: str
    model: str
    response_schema: Optional[Dict[str, Any]]
    json_mode: bool
    temperature: float
    allowed_values: Optional[List[Any]] = None
    result_key: Optional[str] = None
    raw_resp: str = ""
    result: Any = None
    log: Optional[LmLog] = None

    def __iter__(self) -> Iterator[str]:
        raw_resp = ""
        stream = apis.generate_streaming(
            model=self.model,
            prompt=self.prompt,
            response_schema=self.response_schema,
            json_mode=self.json_mode,
            temperature=self.temperature,
            disable_recitation=True,
            disable_safety_check=True,
            stream=True,
        )
        for chunk in stream:
            raw_resp += chunk
            yield chunk

        self.raw_resp = raw_resp
        if self.json_mode:
            parsed = utils.parse_json(raw_resp)
        else:
            parsed = raw_resp

        if parsed and self.result_key:
            parsed = parsed.get(self.result_key)

        self.result = parsed
        self.log = LmLog(prompt=self.prompt, raw_resp=raw_resp, result=parsed)


def format_prompt(prompt_template, worldstate) -> str:
    return jinja2.Template(prompt_template).render(worldstate)


def generate(
    prompt_template: str,
    response_schema: Dict[str, Any],
    worldstate: Dict[str, Any],
    model: str,
    temperature: float = 1.0,
    allowed_values: Optional[List[Any]] = None,
    result_key: Optional[str] = None,
) -> tuple[Any, LmLog]:
    """Generates text from the language model and parses the result.

    Args:
        prompt_template: The Jinja template for the prompt.
        response_schema: The schema for the expected response.
        worldstate: The world state to be rendered into the prompt.
        model: The language model to use.
        temperature: The sampling temperature for the language model.
        allowed_values: An optional list of allowed values for the result. If
          provided, the generation will retry until a result within the allowed
          values is obtained.
        result_key: An optional key to extract a specific value from the parsed
          result. If not provided, the entire parsed result is returned.

    Returns:
        A tuple containing the result (or None if unsuccessful) and the LmLog.
    """

    prompt = format_prompt(prompt_template, worldstate)
    raw_responses = []
    for _ in range(RETRIES):
        raw_resp = None
        try:
            raw_resp = apis.generate(
                model=model,
                prompt=prompt,
                response_schema=response_schema,
                temperature=temperature,
                disable_recitation=True,
                disable_safety_check=True,
            )
            result = utils.parse_json(raw_resp)
            log = LmLog(prompt=prompt, raw_resp=raw_resp, result=result)

            if result and result_key:
                result = result.get(result_key)

            if allowed_values is None or result in allowed_values:
                return result, log

        except Exception as e:
            print(f"Retrying due to Exception: {e}")
        temperature = min(1.0, temperature + 0.2)

        if isinstance(raw_resp, str):
            raw_responses.append(raw_resp)

    return None, LmLog(
        # prompt=prompt, raw_resp="-------".join(raw_responses), result=None
        prompt=prompt, raw_resp=" ||| ".join(raw_responses), result=None
    )

def generate_streaming(
    prompt_template: str,
    response_schema: Dict[str, Any],
    worldstate: Dict[str, Any],
    model: str,
    temperature: float = 1.0,
    allowed_values: Optional[List[Any]] = None,
    result_key: Optional[str] = None,
) -> StreamingGeneration:
    """Returns a streaming wrapper for the language model response.

    Args:
        prompt_template: The Jinja template for the prompt.
        response_schema: The schema for the expected response.
        worldstate: The world state to be rendered into the prompt.
        model: The language model to use.
        temperature: The sampling temperature for the language model.
        allowed_values: An optional list of allowed values for the result. If
          provided, the generation will retry until a result within the allowed
          values is obtained.
        result_key: An optional key to extract a specific value from the parsed
          result. If not provided, the entire parsed result is returned.

    Returns:
        A streaming wrapper that yields chunks and exposes the final log after
        iteration completes.
    """

    prompt = format_prompt(prompt_template, worldstate)
    return StreamingGeneration(
        prompt=prompt,
        model=model,
        response_schema=response_schema,
        json_mode=response_schema is not None,
        temperature=temperature,
        allowed_values=allowed_values,
        result_key=result_key,
    )
