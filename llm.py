from typing import Literal
import requests
import json
import os
import re

class LLMException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

class BadAPIException(LLMException):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg
        
class BadResponseException(LLMException):
    def __init__(self, msg: str = "Bad response in invalid structure"):
        super().__init__(msg)
        self.msg = msg
        
class MissingConfigException(LLMException):
    def __init__(self, msg: str="Missing config file: config.json"):
        super().__init__(msg)
        self.msg = msg

class OpenRouterLLM:
    def __init__(self, brand: str, model: str, free: bool = True, timeout: tuple[float, float]=(30, 200)):
        self.brand = brand
        self.model = model
        self.free = free
        self.model_url = self._make_model_url()
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.key_url = "https://openrouter.ai/api/v1/keys"
        self._key = None
        self._key_hash = None
        self.provision_key = None
        self.timeout=timeout

    def _make_model_url(self) -> str:
        suffix = ":free" if self.free else ""
        return f"{self.brand}/{self.model}{suffix}"

    def _load_provision_key(self):
        try:
            with open('./config.json', 'r') as f:
                data = json.load(f)
                self.provision_key = data['OPEN_ROUTER_PROVISION_KEY']
        except Exception:
            raise MissingConfigException("Missing or invalid config.json for provision key.")

    def _save_key(self):
        # Save API key and key hash to config.json
        config = {}
        if os.path.exists('./config.json'):
            with open('./config.json', 'r') as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    config = {}
        config['OPEN_ROUTER_API_KEY'] = self._key
        config['OPEN_ROUTER_API_KEY_HASH'] = self._key_hash
        with open('./config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def _create_key(self):
        self._load_provision_key()
        headers = {
            "Authorization": f"Bearer {self.provision_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": "SeEvoAppKey",
            "label": "sekey",
            "limit": 1000  # Optional credit limit
        }
        response = requests.post(self.key_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        self._key = data['key']
        self._key_hash = data['data']['hash']
        # Persist new key and hash
        self._save_key()

    def _delete_key(self):
        if not self._key_hash:
            raise MissingConfigException("No key hash available to delete.")
        headers = {
            "Authorization": f"Bearer {self.provision_key}",
            "Content-Type": "application/json"
        }
        response = requests.delete(f"{self.key_url}/{self._key_hash}", headers=headers)
        response.raise_for_status()
        # Remove from config
        config = {}
        with open('./config.json', 'r') as f:
            config = json.load(f)
        config.pop('OPEN_ROUTER_API_KEY', None)
        config.pop('OPEN_ROUTER_API_KEY_HASH', None)
        with open('./config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def get_key(self, key_hash: str = None):
        # Retrieve an existing API key via its hash
        if key_hash:
            self._key_hash = key_hash
        elif self._key_hash:
            key_hash = self._key_hash
        else:
            # Try loading from config
            try:
                with open('./config.json', 'r') as f:
                    data = json.load(f)
                    key_hash = data.get('OPEN_ROUTER_API_KEY_HASH')
            except Exception:
                pass
        if not key_hash:
            raise MissingConfigException("No key hash provided or found in config.")
        # Load provision key if not already
        if not self.provision_key:
            self._load_provision_key()
        headers = {
            "Authorization": f"Bearer {self.provision_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.key_url}/{key_hash}", headers=headers)
        response.raise_for_status()
        data = response.json()
        self._key = data['key']
        self._key_hash = key_hash
        return self._key

    def get_response(self, prompt: str):
        if not self._key:
            # Attempt to load existing key
            try:
                with open('./config.json', 'r') as f:
                    data = json.load(f)
                    self._key = data.get('OPEN_ROUTER_API_KEY')
                    self._key_hash = data.get('OPEN_ROUTER_API_KEY_HASH')
            except Exception:
                pass
            if not self._key:
                # Create new key if missing
                self._create_key()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            'model': self.model_url,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'safety': 'ON'
        })
        
        try:
            response = requests.post(url=self.url, headers=headers, data=data, timeout=self.timeout)
            response.raise_for_status()  # Kiểm tra lỗi HTTP (4xx, 5xx)
            print(response.text)
            if 'choices' not in response.json():
                raise BadResponseException()
            elif 'error' in response.json():
                raise BadAPIException(msg=response.json()['error']['message'])
            return response.json()['choices'][0]['message']['content']
        except requests.Timeout:
            raise BadAPIException("Timeout Request!")
        except requests.RequestException as e:
            raise BadAPIException(str(e))
        
    def extract_response(self, response: str) -> dict:
        m = re.search(r'(json)?(?P<obj>[^\`]+)', response)
        if m is not None:
            json_str = m.group('obj')
            
            try:
                json_obj = json.loads(json_str)
            
                return json_obj
            except json.decoder.JSONDecodeError as e:
                raise BadResponseException(msg="Bad response:" + json_str[:10])
            except TypeError as e:
                raise BadResponseException(msg="Bad response:" + json_str[:10])
        else:
            raise BadResponseException()
        
        
    def close(self):
        self._delete_key()