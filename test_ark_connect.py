import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ARK_API_KEY")
ep_id = os.getenv("ARK_EP_ID")

client = OpenAI(
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    timeout=20   # 超时时间拉长到20秒
)

try:
    resp = client.chat.completions.create(
        model=ep_id,
        messages=[{"role":"user","content":"你好，只回复ok"}],
        max_tokens=10
    )
    print("✅ API连通成功！返回：", resp.choices[0].message.content)
except Exception as e:
    print("❌ API调用失败")
    print(type(e).__name__, str(e))
