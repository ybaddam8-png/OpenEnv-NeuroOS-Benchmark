import os, httpx, time
from openai import OpenAI

API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN")

transport = httpx.HTTPTransport(retries=2, local_address="0.0.0.0")
http_client = httpx.Client(transport=transport, timeout=httpx.Timeout(90.0, connect=10.0))
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, http_client=http_client)

big_prompt = (
    "You are a Neuro-Inclusive UI Auditor.\n"
    "Nodes: [{id:n1,contrast:2.1,aria_label:null,font_size:12},{id:n2,contrast:5.0,aria_label:Header,font_size:18}]\n"
    "Output ONLY valid JSON with an actions array.\n"
    'Example: {"actions": [{"op": "set_contrast", "node_id": "n1", "value": 4.5}]}'
)

for i in range(5):
    t = time.time()
    try:
        r = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": big_prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        elapsed = time.time() - t
        content = r.choices[0].message.content or ""
        print(f"Attempt {i+1}: OK in {elapsed:.1f}s — {content[:100]}")
    except Exception as e:
        elapsed = time.time() - t
        print(f"Attempt {i+1}: FAILED in {elapsed:.1f}s — {type(e).__name__}: {str(e)[:200]}")