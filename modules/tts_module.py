from gtts import gTTS
import os
import uuid
import time

def clean_old_tts_files(out_dir: str = 'tts_output', max_age_sec: int = 60*60*48):
    """
    out_dir 내 2일(48시간) 이상 지난 mp3 파일을 삭제
    """
    now = time.time()
    if not os.path.exists(out_dir):
        return
    for fname in os.listdir(out_dir):
        if fname.endswith('.mp3'):
            fpath = os.path.join(out_dir, fname)
            try:
                if now - os.path.getmtime(fpath) > max_age_sec:
                    os.remove(fpath)
            except Exception:
                pass

def text_to_speech(text: str, lang: str = 'ko', out_dir: str = 'tts_output') -> str:
    """
    입력된 텍스트를 TTS로 변환하여 mp3 파일로 저장하고, 파일 경로를 반환합니다.
    """
    clean_old_tts_files(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(out_dir, filename)
    tts = gTTS(text=text, lang=lang)
    tts.save(filepath)
    return filepath

# 사용 예시
if __name__ == "__main__":
    sample_text = "안녕하세요. 이것은 TTS 테스트입니다."
    mp3_path = text_to_speech(sample_text)
    print(f"생성된 mp3 파일: {mp3_path}") 