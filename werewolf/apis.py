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

from openai import OpenAI
import os

from typing import Any


def generate(model, **kwargs):
    if "KTO" in model or "SFT" in model:
        return generate_kto(model, **kwargs)
    else:
        return generate_openai(model, **kwargs)


# openai
def generate_openai(model: str, prompt: str, json_mode: bool = True, response_schema: dict = None, **kwargs) -> str:
    
    port = os.getenv("API_PORT", "8000")
    client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{port}/v1",
    )
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
            response_format = {"type": "json_object" }
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        model=model,
        max_tokens=2048,
        n=1
    )

    txt = response.choices[0].message.content
    return txt

def generate_kto(model: str, prompt: str, json_mode: bool = True, response_schema: dict = None, **kwargs) -> str:
    
    if "KTO" in model:
        port = os.getenv("KTO_API_PORT", "9001")
    else:
        port = os.getenv("API_PORT", "9000")
    client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{port}/v1",
    )
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
            response_format = {"type": "json_object" }
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        model=model,
        max_tokens=2048,
        n=1
    )

    txt = response.choices[0].message.content
    return txt
