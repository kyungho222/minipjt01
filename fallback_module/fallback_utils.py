# fallback_utils.py
# fallback 분석/생성 관련 함수 모듈
# (의존 모델: AnalysisResponse, DetailedAnalysisResponse, FinalAnalysisResponse 등은 api.py에서 import 필요)
import random
import re
from typing import List, Dict

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
    selected = fallback_questions.copy()
    random.shuffle(selected)
    if count <= len(selected):
        return selected[:count]
    else:
        result = selected[:]
        while len(result) < count:
            random.shuffle(fallback_questions)
            result.extend(fallback_questions[:count - len(result)])
        return result[:count]

def analyze_tf_tendency(text: str) -> float:
    """
    텍스트를 분석하여 T/F 성향 점수를 반환합니다.
    0에 가까울수록 T, 100에 가까울수록 F 성향입니다.
    사고형(T) 무심/단정/객관적 표현이 감지되면 무조건 T로 분류하고, 강도에 따라 점수를 자동 결정합니다.
    싸가지 없는(공감 없는 퉁명/무심) 답변은 T로 살짝 치우치게 점수화합니다.
    """
    text = text.lower()
    final_score = 50  # 모든 경로에서 final_score가 정의되도록 기본값 할당

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
        r"몰라", r"딱히", r"별 생각 없어", r"신경 안 써", r"관심 없어", r"그냥 그래", r"글쎄", r"음[.\.,!\?…]*$", r"별로야",
        r"별다른 이유는 없어", r"딱히 이유 없어", r"그냥 배고파서", r"그냥 그렇다"
    ]
    t_rude_count = sum(len(re.findall(pattern, text)) for pattern in t_rude_patterns)
    if t_rude_count > 0:
        score = max(35, 45 - (t_rude_count - 1) * 3)
        return float(score)

    if re.search(r'^그냥', text) or re.search(r'이유는 없어', text):
        final_score = max(final_score - 10, 15)
    if re.search(r'(결과|성과)(가)?( 더)? (중요|우선|핵심)', text):
        final_score = max(final_score - 12, 15)

    t_keywords_strong = [
        '논리', '분석', '판단', '효율', '객관', '사실', '증거', '합리', '이성', '체계',
        '정확', '명확', '일관', '데이터', '통계', '측정',
        '맞다', '틀렸다', '정답', '확실', '명백', '분명', '확인',
        '검토', '평가', '기준', '조건', '해결', '개선',
        '최적', '효과', '결정', '선택', '우선순위', '중요도',
        '불가능', '문제', '해답', '답', '반드시', '무조건', '체크', '실용적', '계산',
        '결과', '성과', '해결', '판단', '우선', '핵심'
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

    t_count = sum(2 for keyword in t_keywords_strong if keyword in text) + sum(1 for keyword in t_keywords_weak if keyword in text)
    f_count = sum(2 for keyword in f_keywords_strong if keyword in text) + sum(1 for keyword in f_keywords_weak if keyword in text)

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
    else:
        keyword_weight = 0.4
        pattern_weight = 0.25
        tone_weight = 0.2
        structure_weight = 0.15
    
    final_score = (
        keyword_score * keyword_weight +
        pattern_score * pattern_weight +
        tone_score * tone_weight +
        structure_score * structure_weight
    )
    # 길이가 매우 짧으면(10자 이하) 점수 중립화
    if text_length <= 10:
        final_score = 50
    return float(round(final_score, 1))

def get_f_friendly_alternatives():
    return [
        "야 우리 대화 어쩌다 이렇게 정서의 끝으로 갔냐?",
        "지금 감정선 너무 타이트한데, 이거 끊으면 우냐?",
        "우리 지금 말 안 하고 있어도 서로 감정 3줄 요약 중이지?",
        "잠깐만, 나 너 감정까지 느끼고 있거든? 약간 벅차.",
        "와 나 지금 너 대신 감정 써주고 있는 기분인데?",
        "우리 지금 공감으로 대화하는 거야, 단어는 부사야.",
        "이제 우리 대화도 감정 배터리 달아서 충전 좀 해야 될 듯.",
        "지금 너 얘기하는데 내 마음이 더 바빠. 이거 맞냐?",
        "감정 넘치는 거 좋은데, 지금 방 안에 습도 몇이야?",
        "야 감정 너무 나눴다, 이제 조금만 이기적으로 살자."
    ]

def get_t_strong_ment():
    return [
        "뼈 맞았어…",
        "어디보자 데스노트가...",
        "복수한다…",
        "넌 진짜 감정이란 게 있니?",
        "넌 D졌다",
        "조만간 순살만들어준다",
        "로봇이냐..?",
        "유 스틸 마이 데스노트 넘버원~",
        "우리 헤어져",
        "저리가 ㅠㅠ"
    ]

def get_t_mild_ment():
    return [
        "계산기냐?",
        "님 배려좀...",
        "로봇이냐?",
        "살살해주세요..",
        "니 말도 맞는데.. 살살좀 ㅠ",
        "내 기분 존중좀 ㅠ",
        "말 대신 결과?",
        "감정도 좀 챙기라구!",
        "네 논리 따라가다 머리 터져 죽겠어",
        "팩트부터 정리해라? 내 마음은 누가 정리해줘?"
    ]

def generate_final_analysis(results: List[Dict]) -> 'FinalAnalysisResponse':
    # 실제 구현은 api.py에서 복사/이동 필요 (api.py에서 FinalAnalysisResponse import 필요)
    pass

def generate_detailed_analysis(question: str, answer: str, score: float):
    # 실제 구현은 api.py에서 복사/이동 필요 (api.py에서 DetailedAnalysisResponse import 필요)
    pass 