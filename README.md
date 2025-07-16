# MBTI T/F 성향 분석기

이 프로젝트는 사용자의 텍스트 응답을 분석하여 MBTI의 T(사고형)/F(감정형) 성향을 판단하는 웹 애플리케이션입니다.

## 기능

- 텍스트 기반 T/F 성향 분석
- 실시간 감정 분석
- 시각적 결과 표현
- 다중 질문 분석 지원 (1회, 3회, 5회)

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 백엔드 서버 실행:
```bash
python api.py
```

3. 프론트엔드 실행:
- 웹 서버를 사용하여 index.html 파일을 호스팅
- 예: Python의 http.server 사용
```bash
python -m http.server 5500
```

4. 웹 브라우저에서 접속:
```
http://localhost:5500
```

## 기술 스택

- Frontend: HTML, CSS, JavaScript
- Backend: FastAPI (Python)
- AI Model: Hugging Face Transformers 