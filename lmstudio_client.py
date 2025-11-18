import requests

class LMStudioClient:
    def __init__(self, base_url="http://localhost:1234/v1"):
        self.base_url = base_url

    def chat_completion(self, messages, temperature=0.7):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": -1,
                "stream": False
            }
        )
        return response.json()