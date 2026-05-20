
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

res = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4-flash",
    messages=[{"role": "user", "content": "Say hello in one sentence"}]
)

print(res.choices[0].message.content)