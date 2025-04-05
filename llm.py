from typing import Literal
import requests
import json
import re

class BadAPIException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
        
class BadResponseException(Exception):
    def __init__(self, msg: str = "Bad response in invalid structure"):
        super().__init__()
        self.msg = msg

class OpenRouterLLM:
    def __init__(self, brand: str, model: str, free: bool=True):
        self.brand = brand
        self.model = model
        self.free = free
        self.model_url = self._make_model_url()
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.key_url = "https://openrouter.ai/api/v1/keys"
        with open('./config.json', 'r') as f:
            data = json.load(f)
        
        self.provision_key = data['OPEN_ROUTER_PROVISION_KEY']

        self._key = None
        self._key_hash = None
        
        self._create_key()
        
    def _make_model_url(self) -> str:
        return self.brand + "/" + self.model + (":free" if self.free else "")
        
    def _create_key(self):
        headers={
            "Authorization": f"Bearer {self.provision_key}",
            "Content-Type": "application/json"
        }
        json = {
            "name": "SeEvoAppKey",
            "label": "sekey",
            "limit": 1000  # Optional credit limit
        }
        response = requests.post(url=f'{self.key_url}', headers=headers, json=json)
        self._key = response.json()['key']
        print(self._key)
        self._key_hash = response.json()['data']['hash']
    
    def _delete_key(self):
        response = requests.delete(
            url=f'{self.key_url}/{self._key_hash}',
            headers={
                "Authorization": f"Bearer {self.provision_key}",
                "Content-Type": "application/json"
            }
        )
        
    def get_model(self):
        if self.model_url == 'deepseek-v3-0324':
            return 'deepseek/deepseek-chat-v3-0324:free'
        if self.model_url == 'deepseek-r1-zero':
            return 'deepseek/deepseek-r1-zero:free'
        if self.model_url == 'gemini-2.5-pro-exp':
            return 'google/gemini-2.5-pro-exp-03-25:free'
        return self.model_url
    
    def get_response(self, prompt: str, timeout: float|tuple[float] = 100.0):
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            'model': self.get_model(),
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        })
        
        try:
            response = requests.post(url=self.url, headers=headers, data=data, timeout=timeout)
            response.raise_for_status()  # Kiểm tra lỗi HTTP (4xx, 5xx)
            
            if 'choices' not in response.json():
                raise BadResponseException()
            if 'error' in response.json():
                raise BadAPIException(msg=response.json()['error']['message'])
            return response.json()['choices'][0]['message']['content']
        except requests.Timeout:
            raise BadAPIException("Timeout Request!")
        except requests.RequestException as e:
            raise BadAPIException(str(e))
        
    def extract_repsonse(self, response: str) -> dict:
        m = re.search(r'(json)?(?P<obj>[^\`]+)', response)
        if m is not None:
            json_str = m.group('obj')
            
            json_obj = json.loads(json_str)
            
            return json_obj
        else:
            raise BadResponseException()
        
    def close(self):
        self._delete_key()
    