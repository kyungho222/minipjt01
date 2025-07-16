from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# from transformers.pipelines import pipeline
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import torch
import re
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 초기화
print("Starting MBTI T/F Analyzer...")

# AI 모델 초기화
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    AI_MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')
    print("Gemini AI 모델 초기화 완료", flush=True)
else:
    AI_MODEL = None
    print("⚠️ GEMINI_API_KEY가 설정되지 않음. AI 질문 생성이 비활성화됩니다.", flush=True)

# 감정 분석 파이프라인 - 간단한 키워드 기반으로 대체
# sentiment_classifier = pipeline(
#     "text-classification",
#     model="nlptown/bert-base-multilingual-uncased-sentiment",
#     device=-1  # CPU 사용
# )

class TextRequest(BaseModel):
    text: str

class DetailedAnalysisRequest(BaseModel):
    question: str
    answer: str
    score: float

class AnalysisResponse(BaseModel):
    score: float  # 기존 T/F 점수

class DetailedAnalysisResponse(BaseModel):
    detailed_analysis: str
    reasoning: str
    suggestions: List[str]
    alternative_response: str

class FinalAnalysisRequest(BaseModel):
    results: List[Dict]  # [{question, answer, score}, ...]

class FinalAnalysisResponse(BaseModel):
    overall_tendency: str
    personality_analysis: str
    communication_strategy: str
    strengths: List[str]
    growth_areas: List[str]
    keyword_analysis: Dict[str, Dict[str, int]]  # 카테고리별 키워드 사용 횟수

class QuestionGenerationRequest(BaseModel):
    count: Optional[int] = 5  # 생성할 질문 개수
    difficulty: Optional[str] = "medium"  # easy, medium, hard

async def generate_ai_questions_real(count: int = 5, difficulty: str = "medium") -> List[str]:
    """
    실제 AI를 사용하여 T/F 성향 분석을 위한 질문들을 동적으로 생성합니다.
    """
    if not AI_MODEL:
        print("❌ AI 모델이 초기화되지 않음. 폴백 질문 사용.")
        return generate_fallback_questions(count)
    
    try:
        # 난이도별 프롬프트 설정
        difficulty_prompts = {
            "easy": "일상적이고 가벼운 상황에서",
            "medium": "약간 복잡하고 고민이 필요한 상황에서", 
            "hard": "복잡하고 어려운 딜레마 상황에서"
        }
        
        difficulty_desc = difficulty_prompts.get(difficulty, difficulty_prompts["medium"])
        
        prompt = f"""
        MBTI T/F 성향을 분석하기 위한 상황 질문을 {count}개 생성해줘.

        요구사항:
        1. {difficulty_desc} 어떻게 대응할지 묻는 질문
        2. 관계, 소통, 갈등 해결, 의사결정 상황 중심
        3. T성향(논리적/객관적)과 F성향(감정적/관계중심) 구분이 가능한 상황
        4. 각 질문은 "당신이 어떤 방식으로 어떻게 대응할지 구체적으로 설명해주세요" 형태로 끝나야 함
        5. 실제 일상에서 겪을 수 있는 현실적인 상황들
        6. 한국어로 작성, 존댓말 사용

        예시 형태:
        "친구가 '요즘 너무 힘들어'라고 털어놓았습니다. 당신이 어떤 마음으로 어떻게 대응할지 구체적으로 설명해주세요."

        {count}개의 서로 다른 상황 질문을 생성해줘. 각 질문은 번호 없이 줄바꿈으로 구분해줘.
        """
        
        response = await asyncio.to_thread(AI_MODEL.generate_content, prompt)
        questions_text = response.text.strip()
        
        # 생성된 질문을 리스트로 분할
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        # 빈 질문이나 형식이 맞지 않는 질문 필터링
        valid_questions = []
        for q in questions:
            # 번호나 불필요한 문자 제거
            q = re.sub(r'^\d+[\.\)\-\s]*', '', q)
            q = re.sub(r'^[\-\*\•]\s*', '', q)
            q = q.strip()
            
            if len(q) > 20 and '당신이' in q and ('어떻게' in q or '어떤' in q):
                valid_questions.append(q)
        
        # 요청된 개수만큼 반환
        if len(valid_questions) >= count:
            return valid_questions[:count]
        else:
            # 부족하면 폴백 질문으로 채움
            fallback = generate_fallback_questions(count - len(valid_questions))
            return valid_questions + fallback
            
    except Exception as e:
        print(f"❌ AI 질문 생성 실패: {e}")
        return generate_fallback_questions(count)

def generate_fallback_questions(count: int = 5) -> List[str]:
    """
    AI 실패 시 사용할 폴백 질문들
    """
    fallback_questions = [
        "친구가 갑자기 '요즘 너무 스트레스 받아'라고 털어놓았습니다. 당신이 어떤 방식으로 어떻게 대응할지 구체적으로 설명해주세요.",
        "팀 프로젝트에서 의견이 충돌하고 있습니다. 당신이 어떤 접근방식으로 이 상황을 해결할지 자세히 설명해주세요.",
        "친구가 '나 정말 못생겼지?'라고 진지하게 물어봅니다. 당신이 어떤 방식으로 답변할지 구체적으로 설명해주세요.",
        "약속에 늦은 친구가 변명을 계속합니다. 당신이 어떤 마음으로 어떻게 반응할지 자세히 설명해주세요.",
        "회의에서 내 의견이 무시당한 것 같습니다. 당신이 어떤 방식으로 대처할지 구체적으로 설명해주세요."
    ]
    
    import random
    selected = fallback_questions.copy()
    random.shuffle(selected)
    
    # 요청된 개수만큼 반환 (필요하면 중복 허용)
    if count <= len(selected):
        return selected[:count]
    else:
        result = selected[:]
        while len(result) < count:
            random.shuffle(fallback_questions)
            result.extend(fallback_questions[:count - len(result)])
        return result[:count]

def generate_ai_questions(count: int = 5, difficulty: str = "medium") -> List[str]:
    """
    동기 래퍼 함수 - 기존 호환성 유지
    """
    return asyncio.run(generate_ai_questions_real(count, difficulty))

def analyze_tf_tendency(text: str) -> float:
    """
    텍스트를 분석하여 T/F 성향 점수를 반환합니다.
    0에 가까울수록 T, 100에 가까울수록 F 성향입니다.
    사고형(T) 무심/단정/객관적 표현이 감지되면 무조건 T로 분류하고, 강도에 따라 점수를 자동 결정합니다.
    싸가지 없는(공감 없는 퉁명/무심) 답변은 T로 살짝 치우치게 점수화합니다.
    """
    text = text.lower()
    import re

    # 사고형(T) 강한 무심/단정/객관적 표현 패턴 (확실한 사고형)
    t_strong_patterns = [
        r"어쩌라고", r"상관없어", r"알아서 해", r"내 알 바 아냐", r"그게 나랑 무슨 상관이야",
        r"네 마음대로 해", r"내가 뭘", r"그건 네 문제야", r"그건 중요하지 않아"
    ]
    t_strong_count = sum(len(re.findall(pattern, text)) for pattern in t_strong_patterns)
    if t_strong_count > 0:
        score = max(15, 30 - (t_strong_count - 1) * 5)
        return float(score)

    # 싸가지 없는(공감 없는 퉁명/무심) 패턴 (살짝 T)
    t_rude_patterns = [
        r"몰라", r"딱히", r"별 생각 없어", r"신경 안 써", r"관심 없어", r"그냥 그래", r"글쎄", r"음[.\.\,\!\?…]*$", r"별로야"
    ]
    t_rude_count = sum(len(re.findall(pattern, text)) for pattern in t_rude_patterns)
    if t_rude_count > 0:
        # 퉁명/무심 패턴이 감지되면 35~45점(살짝 T)
        score = max(35, 45 - (t_rude_count - 1) * 3)
        return float(score)

    # 1. 키워드(핵심/약한) 및 가중치
    t_keywords_strong = [
        '논리', '분석', '판단', '효율', '객관', '사실', '증거', '합리', '이성', '체계',
        '정확', '명확', '일관', '데이터', '통계', '측정',
        '맞다', '틀렸다', '정답', '확실', '명백', '분명', '확인',
        '검토', '평가', '기준', '조건', '해결', '개선',
        '최적', '효과', '결정', '선택', '우선순위', '중요도',
        '불가능', '문제', '해답', '답', '반드시', '무조건', '체크', '실용적', '계산'
    ]
    t_keywords_weak = [
        '계획', '전략', '목표', '성과', '방법', '해야', '해야지', '하자', '됐다', '안 돼', '안 됨',
        '확실히', '분명히', '정확히', '당연히', '바로', '먼저', '우선', '일단', '정리', '효과적', '효율적',
        '간단', '복잡', '가능', '됐다', '우선', '일단', '편해', '편리', '쉽다', '어렵다', '시간', '가격', '비용'
    ]
    f_keywords_strong = [
        '감정', '마음', '공감', '배려', '이해', '조화', '협력', '관계', '소통', '친밀',
        '가치', '의미', '도덕', '윤리', '지원', '격려', '행복', '슬프', '걱정', '미안', '고마', '소중', '사랑',
        '따뜻', '포근', '아늑', '편안', '안심', '든든', '기분', '느낌', '마음가짐', '심정',
        '함께', '같이', '서로', '우리 모두', '친구', '가족', '사람들', '동료들',
        '예뻐', '귀여워', '착해', '멋져', '좋아해', '싫어해'
    ]
    f_keywords_weak = [
        '기뻐', '즐거워', '신나', '행복해', '만족', '뿌듯', '속상', '짜증', '화나', '답답', '불안', '신경 쓰여',
        '우리', '다함께', '함께 하자', '같이 하자', '마음에', '따뜻', '포근', '보고 싶어', '만나고 싶어', '하고 싶어'
    ]

    # 가중치 적용 카운트
    t_count = sum(2 for keyword in t_keywords_strong if keyword in text) + sum(1 for keyword in t_keywords_weak if keyword in text)
    f_count = sum(2 for keyword in f_keywords_strong if keyword in text) + sum(1 for keyword in f_keywords_weak if keyword in text)

    # 패턴/어조/구조 분석(기존과 동일)
    f_patterns = [
        r'어떻게 생각|어떤 느낌|괜찮을까|어떨까|좋을 것 같|나쁠 것 같',
        r'하면 좋겠|했으면|인 것 같|느낌이|기분이',
        r'함께|같이|서로|우리|모두|다함께',
        r'미안|고마워|사랑|소중|아껴|챙기',
        r'공감|이해|위로|격려|응원',
        r'좋아|싫어|예뻐|귀여워|재밌|지루',
        r'기분 좋|느낌 좋|마음에|따뜻|포근',
        r'하고 싶어|가고 싶어|보고 싶어|만나고 싶어',
        r'같이 하자|함께 하자|우리 모두|다 같이'
    ]
    t_patterns = [
        r'해야 한다|해야지|하자|하면 돼|되면|안 되면',
        r'당연히|정확히|맞다|틀렸다|옳다|그르다|확실히|분명히',
        r'효율적|체계적|논리적|합리적|객관적',
        r'중요한 건|핵심은|문제는|해결책은|방법은',
        r'먼저|우선|차례로|단계별로|계획적으로',
        r'그냥|바로|빨리|즉시|일단|우선',
        r'안 돼|안 됨|되네|됐다|가능|불가능',
        r'쉽다|어렵다|간단|복잡|편해|편리',
        r'계산|비용|가격|시간|효과|실용'
    ]
    f_pattern_count = sum(len(re.findall(pattern, text)) for pattern in f_patterns)
    t_pattern_count = sum(len(re.findall(pattern, text)) for pattern in t_patterns)
    
    soft_tone = len(re.findall(r'것 같아|인 듯|아마|혹시|면 어떨까|하면 좋겠|~인가|~할까|~지 않을까', text))
    firm_tone = len(re.findall(r'반드시|무조건|확실히|당연히|명백히|분명히|해야|하자|된다|안 된다', text))
    question_suggestion = len(re.findall(r'\?|할까|어떨까|좋을까|어때|괜찮을까', text))
    statement_command = len(re.findall(r'다\.|이다\.|하자\.|해야\.|된다\.|안 된다\.', text))
    
    total_keywords = t_count + f_count
    total_patterns = f_pattern_count + t_pattern_count  
    total_tone = soft_tone + firm_tone
    total_structure = question_suggestion + statement_command
    
    # 2. 키워드 점수(가중치 반영)
    if total_keywords == 0:
        keyword_score = 50
    else:
        t_ratio = t_count / total_keywords if total_keywords > 0 else 0
        f_ratio = f_count / total_keywords if total_keywords > 0 else 0
        if t_ratio > f_ratio:
            intensity = min(t_count, 4)
            keyword_score = 25 - (intensity * 7)  # 기존 30에서 25로, 강도별 -7점씩
        elif f_ratio > t_ratio:
            intensity = min(f_count, 4)
            keyword_score = 75 + (intensity * 7)  # 기존 70에서 75로, 강도별 +7점씩
        else:
            keyword_score = 50
    
    # 3. 패턴/어조/구조 점수(동점/애매시 가중치 증가)
    if total_patterns == 0:
        pattern_score = 50
    else:
        t_pattern_ratio = t_pattern_count / total_patterns
        f_pattern_ratio = f_pattern_count / total_patterns
        if t_pattern_ratio > f_pattern_ratio:
            intensity = min(t_pattern_count, 3)
            pattern_score = 30 - (intensity * 5)
        elif f_pattern_ratio > t_pattern_ratio:
            intensity = min(f_pattern_count, 3)
            pattern_score = 70 + (intensity * 5)
        else:
            pattern_score = 50
    
    if total_tone == 0:
        tone_score = 50
    else:
        if firm_tone > soft_tone:
            intensity = min(firm_tone, 2)
            tone_score = 25 - (intensity * 5)
        elif soft_tone > firm_tone:
            intensity = min(soft_tone, 2)
            tone_score = 75 + (intensity * 5)
        else:
            tone_score = 50
    
    if total_structure == 0:
        structure_score = 50
    else:
        if statement_command > question_suggestion:
            structure_score = 30
        elif question_suggestion > statement_command:
            structure_score = 70
        else:
            structure_score = 50
    
    text_length = len(text.replace(' ', ''))
    # 동점/애매할수록 패턴/어조/구조 가중치 증가
    if total_keywords == 0 or abs(t_count - f_count) <= 2:
        keyword_weight = 0.25
        pattern_weight = 0.3
        tone_weight = 0.25
        structure_weight = 0.2
    elif text_length < 15:
        keyword_weight = 0.5
        pattern_weight = 0.2  
        tone_weight = 0.15
        structure_weight = 0.15
    elif text_length < 30:
        keyword_weight = 0.45
        pattern_weight = 0.25
        tone_weight = 0.15
        structure_weight = 0.15
    elif text_length < 60:
        keyword_weight = 0.4
        pattern_weight = 0.3
        tone_weight = 0.2
        structure_weight = 0.1
    else:
        keyword_weight = 0.35
        pattern_weight = 0.35
        tone_weight = 0.25
        structure_weight = 0.05
    
    final_score = (keyword_score * keyword_weight + 
                   pattern_score * pattern_weight + 
                   tone_score * tone_weight +
                   structure_score * structure_weight)
    
    # 강한 키워드 보너스(기존 유지)
    strong_t_words = ['당연', '확실', '맞다', '틀렸', '해야', '명백', '분명', '확실히']
    strong_f_words = ['사랑', '소중', '배려', '공감', '마음', '감정']
    strong_t = sum(1 for word in strong_t_words if word in text)
    strong_f = sum(1 for word in strong_f_words if word in text)
    if strong_t > strong_f and strong_t > 0:
        bonus = min(strong_t * 3, 8)
        final_score = max(final_score - bonus, 20)
    elif strong_f > strong_t and strong_f > 0:
        bonus = min(strong_f * 3, 8)
        final_score = min(final_score + bonus, 80)
    
    return min(max(final_score, 15), 85)

def generate_f_friendly_response(question: str, answer: str, score: float) -> str:
    """
    F 성향 상대에게 더 효과적인 답변을 제안합니다.
    한 줄 실천 팁(예: '좀 더 친절하게 대하세요!')을 맨 위에, 그 아래 대안 답변을 노출합니다.
    """
    import random
    # 실천 팁 pool (구어체+하이라이트 적용)
    def highlight_tip(tip):
        return f"<span style='font-size:1.2em;color:#ff6600;font-weight:bold'>{tip}</span>"
    f_tips_strong = [
        highlight_tip("상대방의 입장에서 한 번 더 생각해보면 어떨까요?"),
        highlight_tip("상대가 힘들어할 때 먼저 공감의 말을 건네보세요."),
        highlight_tip("상대의 감정을 먼저 인정해주는 한마디가 큰 힘이 됩니다."),
        highlight_tip("상대가 내 말을 듣고 어떤 기분일지 상상해보세요."),
        highlight_tip("상대방이 위로받을 수 있도록 따뜻하게 표현해보세요."),
        highlight_tip("상대의 마음을 헤아리는 말 한마디가 필요할 때입니다."),
        highlight_tip("상대가 원하는 게 뭔지 직접 물어보는 것도 좋아요."),
        highlight_tip("상대가 조언을 원하지 않을 때는, 그냥 들어주는 것만으로도 충분해요."),
        highlight_tip("상대가 실수했을 때는 '괜찮아, 누구나 그럴 수 있어'라고 말해보세요."),
        highlight_tip("상대가 기뻐할 때는 함께 기뻐해 주세요."),
        highlight_tip("상대가 조용할 때는 억지로 말시키지 말고 기다려 주세요."),
        highlight_tip("상대가 고민을 털어놓으면, '네가 그렇게 느끼는 게 이해돼'라고 공감해 주세요."),
        highlight_tip("상대가 화났을 때는 바로 조언하지 말고, 감정을 먼저 받아주세요."),
        highlight_tip("상대가 슬퍼할 때는 '힘들었겠다' 한마디가 큰 힘이 됩니다."),
        highlight_tip("상대가 불안해할 때는 '네가 걱정하는 게 뭔지 말해줄래?'라고 물어보세요."),
        highlight_tip("상대가 기분이 좋아 보이면, '오늘 표정이 밝아 보여서 나도 기분이 좋아'라고 말해보세요."),
        highlight_tip("상대가 힘들어할 때는 '내가 옆에 있어줄게'라고 말해보세요."),
        highlight_tip("상대가 조용히 있고 싶어할 때는, 배려해 주세요."),
        highlight_tip("상대가 감정을 표현할 때는, '네가 그렇게 느끼는 거 정말 중요해'라고 말해보세요."),
        highlight_tip("상대가 고민을 말할 때는, '네가 말해줘서 고마워'라고 해보세요."),
        highlight_tip("상대가 실수해도 괜찮아, 나도 그런 적 있어'라고 공감해 주세요.")
    ]
    f_tips_neutral = [
        highlight_tip("조금 더 부드럽게 표현해보면 상대가 더 편안해할 거예요."),
        highlight_tip("상대의 감정도 함께 고려해보면 좋겠어요."),
        highlight_tip("좀 공감 좀 해줘!"),
        highlight_tip("상대방의 기분을 한 번 더 생각해보면 좋을 것 같아요.")
    ]
    f_tips_weak = [
        highlight_tip("이미 충분히 공감하고 계세요!"),
        highlight_tip("지금처럼만 해도 좋아요!"),
        highlight_tip("상대방과 감정을 나누는 태도가 멋져요!")
    ]
    # 점수에 따라 팁 선택
    if score < 30:
        tip = random.choice(f_tips_strong)
    elif score < 60:
        tip = random.choice(f_tips_neutral)
    else:
        tip = random.choice(f_tips_weak)

    # 대안 답변 pool (50개, 다양한 상황/패턴/공감/격려/경청/인정/기쁨 등)
    alternative_pool = [
    f"{answer} 대신, '네가 그렇게 느끼는 게 정말 이해돼. 혹시 내가 도울 수 있는 게 있을까?'라고 말해보세요.",
    f"{answer}에 공감의 한마디를 더해보면 상대가 더 편안해할 수 있어요.",
    f"{answer}도 좋지만, '네가 힘들었겠다. 내 얘기도 들어줄래?'라고 대화를 이어가면 더 좋을 것 같아요.",
    f"{answer}에 '내가 네 입장이었어도 비슷하게 느꼈을 것 같아'라는 말을 덧붙여보세요.",
    f"{answer}에 '네가 이렇게 솔직하게 말해줘서 고마워'라고 해보세요.",
    f"{answer}에 '네가 원하는 게 있으면 언제든 말해줘'라고 마무리해보세요.",
    f"{answer}에 '내가 옆에 있어줄게'라는 말을 추가해보세요.",
    f"{answer}에 '네가 힘들 때 언제든 연락해도 돼'라고 해보세요.",
    f"{answer}에 '네가 기뻐하는 모습을 보니 나도 기뻐'라고 말해보세요.",
    f"{answer}에 '네가 실수해도 괜찮아, 나도 그런 적 있어'라고 공감해보세요.",
    f"{answer}에 '네가 내게 고민을 털어놔줘서 고마워'라고 해보세요.",
    f"{answer}에 '네가 원하는 게 있으면 언제든 말해줘'라고 해보세요.",
    f"{answer}에 '네가 힘들 때는 언제든 내게 기대도 돼'라고 말해보세요.",
    f"{answer}에 '네가 내게 솔직하게 말해줘서 고마워'라고 해보세요.",
    f"{answer}에 '네가 힘들 때는 내가 곁에 있어줄게'라고 해보세요.",
    f"{answer}에 '네가 기뻐할 때는 함께 기뻐해줄게'라고 말해보세요.",
    f"{answer}에 '네가 슬플 때는 곁에 있어줄게'라고 해보세요.",
    f"{answer}에 '네가 화날 때는 감정을 먼저 받아줄게'라고 해보세요.",
    f"{answer}에 '네가 불안할 때는 네 걱정을 들어줄게'라고 해보세요.",
    f"{answer}에 '네가 조용히 있고 싶을 때는 기다려줄게'라고 해보세요.",
    f"{answer}에 '네가 감정을 표현할 때는 소중하게 들어줄게'라고 해보세요.",
    f"{answer}에 '네가 고민을 말해줘서 고마워'라고 해보세요."
    ]
    # 대안 답변 템플릿 pool (20개 이상, answer 활용/비활용 혼합, 자연스러운 연결)
    alternative_templates = [
    "상대의 감정을 먼저 인정해주는 한마디가 큰 힘이 됩니다.",
    "상대방이 힘들어할 때는 먼저 공감의 말을 건네보세요.",
    "상대의 입장을 먼저 물어봐 주세요. '네 입장에선 어땠어?'라고요.",
    "상대가 말할 때는 끼어들지 말고 끝까지 들어주세요.",
    "상대가 울거나 화날 때는 조용히 곁에 있어주는 것도 큰 힘이 됩니다.",
    "상대의 감정을 부정하지 말고, '그럴 수도 있지'라고 인정해 주세요.",
    "상대가 원하는 게 뭔지 직접 물어보는 것도 좋아요.",
    "상대가 조언을 원하지 않을 때는, 그냥 들어주는 것만으로도 충분해요.",
    "상대가 실수했을 때는 '괜찮아, 누구나 그럴 수 있어'라고 말해보세요.",
    "상대가 기뻐할 때는 함께 기뻐해 주세요.",
    "상대가 조용할 때는 억지로 말시키지 말고 기다려 주세요.",
    "상대가 고민을 털어놓으면, '네가 그렇게 느끼는 게 이해돼'라고 공감해 주세요.",
    "상대가 화났을 때는 바로 조언하지 말고, 감정을 먼저 받아주세요.",
    "상대가 슬퍼할 때는 '힘들었겠다' 한마디가 큰 힘이 됩니다.",
    "상대가 불안해할 때는 '네가 걱정하는 게 뭔지 말해줄래?'라고 물어보세요.",
    "상대가 기분이 좋아 보이면, '오늘 표정이 밝아 보여서 나도 기분이 좋아'라고 말해보세요.",
    "상대가 힘들어할 때는 '내가 옆에 있어줄게'라고 말해보세요.",
    "상대가 조용히 있고 싶어할 때는, 배려해 주세요.",
    "상대가 감정을 표현할 때는, '네가 그렇게 느끼는 거 정말 중요해'라고 말해보세요.",
    "상대가 고민을 말할 때는, '네가 말해줘서 고마워'라고 해보세요.",
    "상대가 실수해도 괜찮아, 나도 그런 적 있어'라고 공감해 주세요.",
    "상대가 힘들 때는 언제든 내게 기대도 돼'라고 말해보세요.",
    "상대가 내게 솔직하게 말해줘서 고마워'라고 해보세요.",
    "상대가 힘들 때는 내가 곁에 있어줄게'라고 해보세요.",
    "상대가 기뻐할 때는 함께 기뻐해줄게'라고 말해보세요.",
    "상대가 슬플 때는 곁에 있어줄게'라고 해보세요.",
    "상대가 화날 때는 감정을 먼저 받아줄게'라고 해보세요.",
    "상대가 불안할 때는 네 걱정을 들어줄게'라고 해보세요.",
    "상대가 조용히 있고 싶을 때는 기다려줄게'라고 해보세요.",
    "상대가 감정을 표현할 때는 소중하게 들어줄게'라고 해보세요.",
    "상대가 고민을 말해줘서 고마워'라고 해보세요.",
    "상대의 이야기를 충분히 들어주고, 내 생각은 나중에 전해도 늦지 않아요.",
    f"만약 '{answer}'라고 답했다면, 이런 말도 함께 해보면 좋을 것 같아요.",
    f"'{answer}'라고만 하기보다는, 상대의 감정을 먼저 들어주는 것도 좋아요.",
    f"상대의 입장을 먼저 들어주고, 그 다음에 '{answer}'와 같은 의견을 전해보세요.",
    f"'{answer}'에 '내가 네 입장이었어도 비슷하게 느꼈을 것 같아'라는 말을 덧붙여보세요.",
    f"'{answer}'에 '네가 이렇게 솔직하게 말해줘서 고마워'라고 해보세요.",
    f"'{answer}'에 '네가 원하는 게 있으면 언제든 말해줘'라고 마무리해보세요.",
    f"'{answer}'에 '내가 옆에 있어줄게'라는 말을 추가해보세요.",
    f"'{answer}'에 '네가 힘들 때 언제든 연락해도 돼'라고 해보세요.",
    f"'{answer}'에 '네가 기뻐하는 모습을 보니 나도 기뻐'라고 말해보세요.",
    f"'{answer}'에 '네가 실수해도 괜찮아, 나도 그런 적 있어'라고 공감해보세요."
    ]
    # 점수 구간 세분화 (0~100을 템플릿 개수만큼 나눔)
    idx = int(min(max(score, 0), 99) * len(alternative_templates) // 100)
    alternative = alternative_templates[idx % len(alternative_templates)]
    return f"✅ {tip}\n\n{alternative}"

def generate_final_analysis(results: List[Dict]) -> FinalAnalysisResponse:
    """
    전체 질문 결과를 바탕으로 종합적인 T/F 성향 분석과 F 성향 상대 대응법을 제공합니다.
    """
    if not results:
        return FinalAnalysisResponse(
            overall_tendency="분석할 데이터가 없습니다.",
            personality_analysis="",
            communication_strategy="",
            strengths=[],
            growth_areas=[],
            keyword_analysis={}
        )
    
    # 전체 평균 점수 계산
    total_score = sum(r['score'] for r in results) / len(results)
    
    # 점수 분포 분석
    t_responses = sum(1 for r in results if r['score'] < 40)
    neutral_responses = sum(1 for r in results if 40 <= r['score'] <= 60) 
    f_responses = sum(1 for r in results if r['score'] > 60)
    
    # 전체 성향 판단
    if total_score < 30:
        overall_tendency = "강한 T(사고형) 성향"
        tendency_desc = "논리적이고 객관적인 판단을 선호하는"
    elif total_score < 45:
        overall_tendency = "T(사고형) 성향"
        tendency_desc = "합리적 사고를 중시하는"
    elif total_score < 55:
        overall_tendency = "T-F 균형"
        tendency_desc = "논리와 감정의 균형이 잡힌"
    elif total_score < 70:
        overall_tendency = "F(감정형) 성향"
        tendency_desc = "감정과 관계를 중시하는"
    else:
        overall_tendency = "강한 F(감정형) 성향"
        tendency_desc = "깊은 공감과 배려심을 가진"
    
    # 성격 분석
    consistency = 100 - (max([r['score'] for r in results]) - min([r['score'] for r in results]))
    if consistency > 80:
        consistency_desc = "일관성이 매우 높고 안정된"
    elif consistency > 60:
        consistency_desc = "어느 정도 일관된"
    else:
        consistency_desc = "상황에 따라 유연하게 대응하는"
    
    personality_analysis = f"""
당신은 {tendency_desc} {consistency_desc} 성향을 보여주었습니다. 

{len(results)}개의 질문 중 T 성향 답변이 {t_responses}개, 균형적 답변이 {neutral_responses}개, F 성향 답변이 {f_responses}개로 나타났습니다. 
전체적으로 {total_score:.1f}점으로 {overall_tendency}을 나타냅니다.
    """.strip()
    
    # F 성향 상대 대응 전략
    if total_score < 40:  # T 성향이 강한 경우
        communication_strategy = f"""
F 성향 상대와의 효과적인 소통법:

🎯 **핵심 전략**: 논리적 설명 + 감정적 배려

📝 **대화 방식**:
• "객관적으로 보면 이렇습니다만, 당신의 기분은 어떠신가요?"
• "효율적인 방법은 이것이지만, 모두가 편안한 방법을 함께 찾아보죠"
• "사실 분석상으로는... 하지만 팀 분위기도 중요하니까요"

💡 **주의사항**:
• 너무 직설적인 표현보다는 부드러운 어조 사용
• 결론을 먼저 말하기보다 상대방 의견을 먼저 듣기
• "당연히", "확실히" 같은 단정적 표현 자제
        """
    elif total_score < 60:  # 균형적인 경우
        communication_strategy = f"""
F 성향 상대와의 효과적인 소통법:

🎯 **핵심 전략**: 현재의 균형감 활용 + 감정적 표현 강화

📝 **대화 방식**:
• "논리적으로도 맞고 감정적으로도 좋은 방향은..."
• "합리적이면서도 모두가 만족할 수 있는 방법을 찾아보죠"
• "효과적이지만 따뜻한 접근 방식으로 해보면 어떨까요?"

💡 **강화 포인트**:
• 현재의 균형감은 큰 장점 - 이를 잘 활용하세요
• F 성향 상대에게는 감정적 표현을 조금 더 늘려보세요
• "우리 함께", "같이 생각해봐요" 같은 포용적 표현 활용
        """
    else:  # F 성향인 경우
        communication_strategy = f"""
F 성향 상대와의 효과적인 소통법:

🎯 **핵심 전략**: 자연스러운 공감대 형성

📝 **대화 방식**:
• 이미 F 성향이시므로 자연스럽게 잘 소통하고 계십니다
• "마음으로 느끼기에는...", "함께 생각해보면..." 같은 표현이 자연스럽게 나오실 거예요
• 감정적 공감을 바탕으로 한 소통이 편안하실 것입니다

💡 **추가 팁**:
• 가끔 객관적 근거도 함께 제시하면 더욱 설득력 있는 소통 가능
• T 성향이 강한 상대방에게는 논리적 설명을 먼저 하고 감정적 배려를 더하는 방식 시도
        """
    
    # 강점 분석
    hashtag_candidates = {
        'T': ['논리', '객관', '판단', '효율', '일관'],
        'B': ['균형', '융통', '조화', '통합'],
        'F': ['공감', '배려', '관계', '이해', '소통']
    }
    # 점수 구간별 해시태그 선택
    if total_score < 30:
        strengths = hashtag_candidates['T'][:3]  # 논리, 객관, 판단
    elif total_score < 50:
        strengths = hashtag_candidates['T'][:1] + hashtag_candidates['B'][:2]  # 논리, 균형, 융통
    else:
        strengths = hashtag_candidates['F'][:3]  # 공감, 배려, 관계
    
    # 성장 영역
    growth_areas = []
    if total_score < 30:
        growth_areas = [
            "상대방의 감정과 입장 고려하기",
            "부드럽고 따뜻한 표현 방식 연습",
            "논리적 설명과 감정적 배려의 조화",
            "상대방 의견을 먼저 듣는 습관 기르기"
        ]
    elif total_score < 50:
        growth_areas = [
            "F 성향 상대와 소통할 때 감정적 표현 늘리기",
            "공감적 언어 사용 연습",
            "현재의 균형감을 상황에 맞게 조절하기",
            "감정적 니즈에 더 민감하게 반응하기"
        ]
    else:
        growth_areas = [
            "감정적 판단과 함께 객관적 근거 고려하기",
            "때로는 단호한 결정도 필요함을 인식",
            "논리적 설득력 강화",
            "T 성향 상대방과의 소통 방식 다양화"
        ]
    
    # 키워드 분석 - 개선된 버전
    keyword_analysis = {
        'logical_thinking': {},  # 논리적 사고
        'analytical_approach': {},  # 분석적 접근
        'emotional_empathy': {},  # 감정적 공감
        'relationship_focus': {}  # 관계 중심
    }
    
    # 카테고리별 키워드 정의 (확장된 버전 - 짧은 답변을 위한 완화된 기준)
    keyword_categories = {
        'logical_thinking': [
            # 기본 키워드
            '논리', '분석', '판단', '이성', '합리', '객관', '체계', '원리', '일관',
            # 짧은 답변용 추가 키워드
            '맞', '틀', '확실', '당연', '분명', '명확', '정확', '객관적', '논리적', '합리적',
            '사실', '증거', '근거', '이유', '원인', '결과', '방법', '해결'
        ],
        'analytical_approach': [
            # 기본 키워드  
            '효율', '성과', '전략', '계획', '목표', '데이터', '측정', '정확', '명확',
            # 짧은 답변용 추가 키워드
            '계획적', '체계적', '단계', '순서', '먼저', '우선', '중요', '핵심',
            '비교', '평가', '검토', '확인', '선택', '결정', '최적', '효과적'
        ],
        'emotional_empathy': [
            # 기본 키워드
            '감정', '느낌', '마음', '공감', '이해', '위로', '따뜻', '배려',
            # 짧은 답변용 추가 키워드  
            '좋', '싫', '기분', '행복', '슬', '힘들', '걱정', '고민',
            '미안', '고마', '사랑', '소중', '예쁘', '귀여', '재미', '즐거'
        ],
        'relationship_focus': [
            # 기본 키워드
            '관계', '소통', '협력', '조화', '사람', '인간', '도움', '지원', '격려',
            # 짧은 답변용 추가 키워드
            '함께', '같이', '서로', '우리', '친구', '가족', '동료', '팀',
            '배려', '존중', '이해', '도와', '돕', '챙기', '응원'
        ]
    }
    
    # 각 답변에서 키워드 추출 (완화된 매칭 방식)
    for r in results:
        answer_text = r['answer'].lower()  # 소문자로 변환하여 매칭 향상
        
        for category, keywords in keyword_categories.items():
            for keyword in keywords:
                matches = 0
                
                # 1. 정확한 단어 매칭 (기존 방식, 더 높은 가중치)
                pattern = r'\b' + re.escape(keyword) + r'\b'
                exact_matches = len(re.findall(pattern, answer_text, re.IGNORECASE))
                matches += exact_matches * 2  # 정확 매칭은 2배 가중치
                
                # 2. 부분 매칭 (완화된 방식, 짧은 키워드만)
                if len(keyword) <= 2 and exact_matches == 0:  # 2글자 이하 키워드만 부분매칭 허용
                    if keyword in answer_text:
                        matches += 1
                
                # 3. 어근 매칭 (동사/형용사 활용)
                if exact_matches == 0:
                    # 한국어 어근 패턴들
                    root_patterns = {
                        '좋': ['좋아', '좋은', '좋을', '좋다', '좋지', '좋네'],
                        '싫': ['싫어', '싫은', '싫다', '싫네'],
                        '맞': ['맞아', '맞는', '맞다', '맞네', '맞지'],
                        '틀': ['틀려', '틀린', '틀렸'],
                        '도와': ['도와줘', '도와주', '도움'],
                        '돕': ['도와', '도움'],
                        '함께': ['같이'],
                        '확실': ['확실히', '확실한']
                    }
                    
                    if keyword in root_patterns:
                        for variant in root_patterns[keyword]:
                            if variant in answer_text:
                                matches += 1
                                break
                    
                    # 역방향 체크 (키워드가 변형어인 경우)
                    for root, variants in root_patterns.items():
                        if keyword in variants and root in answer_text:
                            matches += 1
                            break
                
                if matches > 0:
                    if keyword not in keyword_analysis[category]:
                        keyword_analysis[category][keyword] = 0
                    keyword_analysis[category][keyword] += matches
    
    return FinalAnalysisResponse(
        overall_tendency=overall_tendency,
        personality_analysis=personality_analysis,
        communication_strategy=communication_strategy,
        strengths=strengths,
        growth_areas=growth_areas,
        keyword_analysis=keyword_analysis
    )

def generate_detailed_analysis(question: str, answer: str, score: float) -> DetailedAnalysisResponse:
    """
    질문, 답변, T/F 점수를 바탕으로 상세한 분석을 생성합니다.
    """
    # T/F 성향 판단
    if score < 30:
        tendency = "T(사고형)"
        tendency_desc = "논리적이고 객관적인"
    elif score < 70:
        tendency = "T와 F의 균형"
        tendency_desc = "논리와 감정의 균형이 잡힌"
    else:
        tendency = "F(감정형)"
        tendency_desc = "감정적이고 공감적인"
    
    # 답변 분석
    answer_length = len(answer)
    if answer_length < 20:
        length_analysis = "간결하고 명확한 표현을 선호하시는 것 같습니다."
    elif answer_length < 50:
        length_analysis = "적절한 길이로 균형잡힌 답변을 하셨습니다."
    else:
        length_analysis = "상세하고 풍부한 표현을 사용하시는 것 같습니다."
    
    # 키워드 기반 세부 분석
    logical_keywords = ['논리', '객관', '사실', '분석', '효율', '합리']
    emotional_keywords = ['감정', '마음', '느낌', '공감', '배려', '이해']
    
    logical_count = sum(1 for keyword in logical_keywords if keyword in answer)
    emotional_count = sum(1 for keyword in emotional_keywords if keyword in answer)
    
    # 상세 분석 생성
    detailed_analysis = f"이 답변에서 당신은 {tendency_desc} 성향을 보여주었습니다. "
    
    if logical_count > emotional_count:
        detailed_analysis += "답변에서 논리적 사고와 객관적 판단을 중시하는 모습이 드러났습니다. "
    elif emotional_count > logical_count:
        detailed_analysis += "답변에서 감정적 공감과 인간관계를 중시하는 모습이 드러났습니다. "
    else:
        detailed_analysis += "답변에서 논리와 감정의 균형을 추구하는 모습이 드러났습니다. "
    
    detailed_analysis += length_analysis
    
    # 근거 설명
    reasoning = f"T/F 점수 {score:.1f}점은 "
    if score < 30:
        reasoning += "강한 T 성향을 나타냅니다. 사실과 논리를 바탕으로 한 판단을 선호하시는 것으로 보입니다."
    elif score < 70:
        reasoning += "T와 F의 균형적 성향을 나타냅니다. 상황에 따라 논리적 판단과 감정적 고려를 모두 활용하시는 것으로 보입니다."
    else:
        reasoning += "강한 F 성향을 나타냅니다. 사람의 감정과 관계를 중시하며 공감적 판단을 선호하시는 것으로 보입니다."
    
    # 제안사항
    suggestions = []
    if score < 30:
        suggestions = [
            "감정적 측면도 고려해보는 연습을 해보세요",
            "다른 사람의 입장에서 생각해보는 시간을 가져보세요",
            "논리적 판단과 함께 인간적 따뜻함도 표현해보세요"
        ]
    elif score < 70:
        suggestions = [
            "현재의 균형잡힌 시각을 잘 활용하고 계십니다",
            "상황에 맞는 적절한 접근법을 선택하는 능력이 뛰어납니다",
            "논리와 감정 모두를 고려하는 통합적 사고를 발전시켜보세요"
        ]
    else:
        suggestions = [
            "감정적 판단과 함께 객관적 근거도 찾아보세요",
            "논리적 분석을 통해 더 나은 결정을 내릴 수 있습니다",
            "공감 능력을 바탕으로 합리적 해결책을 모색해보세요"
        ]
    
    # F 성향 상대를 위한 대안 답변 생성
    alternative_response = generate_f_friendly_response(question, answer, score)
    
    return DetailedAnalysisResponse(
        detailed_analysis=detailed_analysis,
        reasoning=reasoning,
        suggestions=suggestions,
        alternative_response=alternative_response
    )

def log_debug(msg):
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write("[DEBUG] analyze_text 함수 진입!\n")
    log_debug(f"[DEBUG] /analyze 요청 도착, AI_MODEL: {AI_MODEL}")
    try:
        if AI_MODEL:
            log_debug("[DEBUG] Gemini AI 분석 분기 진입")
            try:
                prompt = f"""
                아래 답변을 MBTI T/F(사고형/감정형) 관점에서 분석해줘.\n- T(사고형) 성향이면 'T', F(감정형) 성향이면 'F', 논리와 감정의 균형/중립/밸런스면 'B'라는 키워드를 반드시 포함해서 자연어로 분석해줘.\n답변: {request.text.strip()}\n"""
                response = await asyncio.to_thread(AI_MODEL.generate_content, prompt)
                log_debug(f"[Gemini AI 전체 응답]: {response}")
                result = response.text.strip().upper()
                log_debug(f"[Gemini AI 원본 응답]: {result}")
                # Gemini 응답이 비정상(빈 값, 에러, 쿼터 등)일 때도 fallback
                if (not result) or ("429" in result) or ("QUOTA" in result) or ("ERROR" in result):
                    log_debug("[Gemini 응답 비정상, fallback으로 자체 분석 수행]")
                    tf_score = analyze_tf_tendency(request.text)
                    log_debug("[분석 로직: fallback]")
                else:
                    # 키워드 파싱
                    if 'T' in result and 'F' not in result:
                        tf_score = 20
                    elif 'F' in result and 'T' not in result:
                        tf_score = 80
                    elif any(k in result for k in ['B', '균형', '중립', '밸런스']):
                        tf_score = 50
                    elif 'T' in result and 'F' in result:
                        tf_score = 50
                    else:
                        log_debug(f"[Gemini AI 예외: 예상치 못한 응답] {result}")
                        tf_score = analyze_tf_tendency(request.text)
                        log_debug("[분석 로직: fallback]")
                    log_debug("[분석 로직: gemini]")
            except Exception as e:
                log_debug(f"[Gemini AI 예외 발생, fallback으로 자체 분석 수행]: {e}")
                tf_score = analyze_tf_tendency(request.text)
                log_debug("[분석 로직: fallback]")
        else:
            log_debug("[DEBUG] Fallback(키워드 분석) 분기 진입")
            tf_score = analyze_tf_tendency(request.text)
            log_debug("[분석 로직: fallback]")
        return AnalysisResponse(score=tf_score)
    except Exception as e:
        log_debug(f"[analyze_text 최상위 예외]: {e}")
        tf_score = analyze_tf_tendency(request.text)
        log_debug("[분석 로직: fallback]")
        return AnalysisResponse(score=tf_score)

@app.post("/detailed_analyze")
async def detailed_analyze(request: DetailedAnalysisRequest):
    try:
        if AI_MODEL:
            log_debug("[DEBUG] Gemini AI 상세 분석 분기 진입")
            try:
                prompt = f"""
                아래 질문과 답변을 MBTI T/F(사고형/감정형) 관점에서 상세하게 분석해줘.\n- 분석 결과를 1) 성향 분석(자연어), 2) 근거(자연어), 3) 개선 제안(3개), 4) F 성향 상대를 위한 한 줄 실천 팁(짧고 구체적, [실천팁] 태그), 5) F 성향 상대를 위한 대안 답변(자연어, [대안] 태그)로 구분해서 각각 한글로 출력해줘.\n- 각 항목은 반드시 [분석], [근거], [제안], [실천팁], [대안] 태그로 시작해줘.\n질문: {request.question}\n답변: {request.answer}\n점수: {request.score}\n"""
                response = await asyncio.to_thread(AI_MODEL.generate_content, prompt)
                log_debug(f"[Gemini 상세분석 전체 응답]: {response}")
                result = response.text.strip()
                log_debug(f"[Gemini 상세분석 원본 응답]: {result}")
                # Gemini 응답이 비정상(빈 값, 에러, 쿼터 등)일 때 fallback
                if (not result) or ("429" in result) or ("QUOTA" in result) or ("ERROR" in result):
                    log_debug("[Gemini 상세분석 응답 비정상, fallback으로 자체 분석 수행]")
                    return generate_detailed_analysis(request.question, request.answer, request.score)
                # 파싱
                def extract(tag):
                    import re
                    m = re.search(rf"\[{tag}\](.*?)(?=\[|$)", result, re.DOTALL)
                    return m.group(1).strip() if m else ""
                detailed_analysis = extract("분석")
                reasoning = extract("근거")
                suggestions_raw = extract("제안")
                suggestions = [s.strip("-• ") for s in suggestions_raw.split("\n") if s.strip()] if suggestions_raw else []
                tip = extract("실천팁")
                alternative = extract("대안")
                # alternative_response: 실천팁 + 대안 합쳐서 반환 (fallback과 동일 구조)
                alternative_response = (f"✅ {tip}\n\n{alternative}") if tip or alternative else "Gemini 분석 결과를 받아오지 못했습니다."
                return DetailedAnalysisResponse(
                    detailed_analysis=detailed_analysis or "Gemini 분석 결과를 받아오지 못했습니다.",
                    reasoning=reasoning or "Gemini 분석 결과를 받아오지 못했습니다.",
                    suggestions=suggestions or ["Gemini 분석 결과를 받아오지 못했습니다."],
                    alternative_response=alternative_response
                )
            except Exception as e:
                log_debug(f"[Gemini 상세분석 예외 발생, fallback으로 자체 분석 수행]: {e}")
                return generate_detailed_analysis(request.question, request.answer, request.score)
        else:
            log_debug("[DEBUG] Fallback(키워드 상세 분석) 분기 진입")
            return generate_detailed_analysis(request.question, request.answer, request.score)
    except Exception as e:
        log_debug(f"[detailed_analyze 최상위 예외]: {e}")
        return generate_detailed_analysis(request.question, request.answer, request.score)

@app.post("/final_analyze")
async def final_analyze(request: FinalAnalysisRequest):
    try:
        final_result = generate_final_analysis(request.results)
        return final_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_questions")
async def generate_questions(request: QuestionGenerationRequest):
    """
    AI 기반으로 새로운 질문들을 생성합니다.
    """
    try:
        questions = await generate_ai_questions_real(count=request.count or 5, difficulty=request.difficulty or "medium")
        return {
            "questions": questions,
            "count": len(questions),
            "difficulty": request.difficulty,
            "generated_by": "AI"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질문 생성 중 오류가 발생했습니다: {str(e)}")

@app.get("/questions")
async def get_questions(use_ai: str = "false", count: int = 5, difficulty: str = "medium"):
    """
    질문들을 반환합니다. use_ai=true이면 AI 생성 질문을, false이면 기본 질문을 반환합니다.
    """
    try:
        use_ai_bool = str(use_ai).lower() in ["true", "1", "yes"]
        if use_ai_bool:
            # AI 생성 질문 사용
            questions = await generate_ai_questions_real(count=count, difficulty=difficulty)
            return {
                "questions": questions,
                "source": "AI",
                "count": len(questions),
                "difficulty": difficulty
            }
        else:
            # 기본 질문 파일 사용
            questions_file = Path("question/questions.json")
            if not questions_file.exists():
                # 파일이 없으면 AI 생성 질문으로 대체
                questions = await generate_ai_questions_real(count=5, difficulty="medium")
                return {
                    "questions": questions,
                    "source": "AI_fallback",
                    "count": len(questions),
                    "difficulty": "medium"
                }
            
            with open(questions_file, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
            
            questions_data["source"] = "file"
            return questions_data
            
    except json.JSONDecodeError:
        # JSON 파일 오류 시 AI 생성 질문으로 대체
        questions = await generate_ai_questions_real(count=5, difficulty="medium")
        return {
            "questions": questions,
            "source": "AI_fallback",
            "count": len(questions),
            "difficulty": "medium"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static 파일 마운트 (API 엔드포인트 뒤에 위치)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

@app.post("/reset_log")
async def reset_log():
    with open("debug.log", "w", encoding="utf-8") as f:
        f.write("[DEBUG] 로그가 초기화되었습니다!\n")
    return Response(content="로그 초기화 완료", media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    # 서버 시작 시 debug.log 파일 초기화
    with open("debug.log", "w", encoding="utf-8") as f:
        f.write("[DEBUG] 서버가 실행되었습니다!\n")
        f.write(f"[DEBUG] 실행 중인 파일: {os.path.abspath(__file__)}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000) 