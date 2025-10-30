from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")

resp = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role":"user","content":"Explain cosine similarity simply."}],
)
print(resp.choices[0].message.content)
