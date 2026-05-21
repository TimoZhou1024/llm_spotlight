from openai import OpenAI
import sys

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

MODEL_NAME = "/root/Workspace/CBY/models/base/Qwen3-30B-A3B-Instruct-2507"

def check_vllm():
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a health check bot."},
                {"role": "user", "content": "Reply with OK."}
            ],
            max_tokens=5,
            temperature=0.0,
        )

        text = resp.choices[0].message.content
        print("✅ vLLM service is healthy")
        print("Response:", text)
        return True

    except Exception as e:
        print("❌ vLLM service is NOT healthy")
        print("Error:", repr(e))
        return False


if __name__ == "__main__":
    ok = check_vllm()
    sys.exit(0 if ok else 1)
