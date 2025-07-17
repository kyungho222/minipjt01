import requests

with open('tts_test.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

url = "http://localhost:8000/tts"
data = {
    "text": text,
    "lang": "ko"
}
response = requests.post(url, json=data)
if response.status_code == 200:
    with open("tts_test.mp3", "wb") as f:
        f.write(response.content)
    print("mp3 파일 저장 완료! tts_test.mp3")
else:
    print("에러:", response.status_code, response.text) 