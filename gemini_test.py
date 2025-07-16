import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel('gemini-2.5-pro-latest')
    response = model.generate_content("안녕, 너는 누구야?")
    print("Gemini 응답:", response.text)
except Exception as e:
    print("Gemini API 예외 발생:", e) 