import urllib.request
import json

url = "https://api-inference.huggingface.co/models/premgouda1916/kannada-sentiment-classifier-Xlm_RoBERTa"
data = {"inputs": "ನನಗೆ ತುಂಬಾ ಸಂತೋಷವಾಗಿದೆ"}
headers = {"Content-Type": "application/json"}

req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("Success:")
        print(json.dumps(res, indent=2))
except Exception as e:
    print("Error:", e)
