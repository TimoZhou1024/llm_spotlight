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

import os
from typing import Any, Iterator

from openai import OpenAI


def generate(model = False, **kwargs):

    if "KTO" in model or "SFT" in model:
        return generate_kto(model, **kwargs)
    else:
        return generate_openai(model, **kwargs)

def _get_client_and_base_url(model: str):
    if "KTO" in model:
        port = os.getenv("KTO_API_PORT", "9001")
        return OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    if "Qwen3-4B" in model:
        port = os.getenv("API_PORT", "9000")
        return OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    return OpenAI(
        api_key=os.environ["QWEN_API"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def _get_response_format(json_mode: bool, response_schema: dict = None):
    response_format = {"type": "text"}
    if json_mode:
        if response_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": response_schema,
                    "strict": True
                }
            }
        else:
            response_format = {"type": "json_object"}
    return response_format

def generate_streaming(
    model: str,
    prompt: str,
    json_mode: bool = False,
    response_schema: dict = None,
    **kwargs
) -> Iterator[str]:
 
    client = _get_client_and_base_url(model)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_format=_get_response_format(json_mode, response_schema),
        model=model,
        max_tokens=2048,
        n=1,
        stream=True
    )

    for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        first_choice = choices[0]
        delta = getattr(first_choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content:
            yield content
    


# openai
def generate_openai(model: str, prompt: str, json_mode: bool = True, response_schema: dict = None, **kwargs) -> str:
    client = _get_client_and_base_url(model)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_format=_get_response_format(json_mode, response_schema),
        model=model,
        max_tokens=2048,
        n=1
    )

    txt = response.choices[0].message.content
    return txt

def generate_kto(model: str, prompt: str, json_mode: bool = True, response_schema: dict = None, **kwargs) -> str:
    client = _get_client_and_base_url(model)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_format=_get_response_format(json_mode, response_schema),
        model=model,
        max_tokens=2048,
        n=1
    )

    txt = response.choices[0].message.content
    return txt
