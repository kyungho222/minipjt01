import whisper
import os

# Whisper 모델 로딩 (최초 1회)
whisper_model = whisper.load_model("small")


def transcribe_audio_file(audio_path: str) -> str:
    """
    주어진 오디오 파일 경로를 Whisper로 텍스트 변환하여 반환
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    # 언어를 한국어('ko')로 강제 지정
    result = whisper_model.transcribe(audio_path, language='ko')
    text = result.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    return text 