// API 기본 URL 설정
const API_BASE_URL = 'http://localhost:8000';

// 전역 변수
let currentCount = 0;
let maxCount = 0;
let results = [];
let usedQuestions = new Set();
let questions = [];

// URL 파라미터에서 count와 type 가져오기
function getParamsFromURL() {
    const urlParams = new URLSearchParams(window.location.search);
    return {
        count: parseInt(urlParams.get('count')) || 5,
        type: urlParams.get('type') || 'meme'
    };
}

// 질문 로딩 모달 표시
function showQuestionLoadingModal() {
    const modal = document.getElementById('question_loading_modal');
    if (modal) {
        modal.style.display = 'block';
    }
}

// 질문 로딩 모달 숨기기
function hideQuestionLoadingModal() {
    const modal = document.getElementById('question_loading_modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// 분석 모달 표시
function showAnalysisModal() {
    const modal = document.getElementById('analysis_modal');
    if (modal) {
        modal.style.display = 'block';
    }
}

// 분석 모달 숨기기
function hideAnalysisModal() {
    const modal = document.getElementById('analysis_modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// 마이크 모달 표시
function showMicModal() {
    const modal = document.getElementById('mic_modal');
    if (modal) {
        modal.style.display = 'flex';
        resetMicModal();
    }
}

// 마이크 모달 숨기기
function hideMicModal() {
    const modal = document.getElementById('mic_modal');
    if (modal) {
        modal.style.display = 'none';
        stopRecording();
    }
}

// 마이크 모달 초기화
function resetMicModal() {
    const statusText = document.getElementById('mic_status_text');
    const recognizedResult = document.getElementById('recognized_result');
    const startButton = document.getElementById('start_recording');
    const stopButton = document.getElementById('stop_recording');
    
    if (statusText) statusText.textContent = '마이크 버튼을 눌러 음성 인식을 시작하세요';
    if (recognizedResult) recognizedResult.textContent = '';
    if (startButton) startButton.style.display = 'inline-block';
    if (stopButton) stopButton.style.display = 'none';
}

// 음성 인식 변수
let recognition = null;
let isRecording = false;

// 음성 인식 시작
function startRecording() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('이 브라우저는 음성 인식을 지원하지 않습니다.');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.lang = 'ko-KR';
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onstart = function() {
        isRecording = true;
        const statusText = document.getElementById('mic_status_text');
        const startButton = document.getElementById('start_recording');
        const stopButton = document.getElementById('stop_recording');
        
        if (statusText) statusText.textContent = '음성을 인식하고 있습니다...';
        if (startButton) startButton.style.display = 'none';
        if (stopButton) stopButton.style.display = 'inline-block';
    };
    
    recognition.onresult = function(event) {
        const result = event.results[0][0].transcript;
        const recognizedResult = document.getElementById('recognized_result');
        const statusText = document.getElementById('mic_status_text');
        
        if (recognizedResult) recognizedResult.textContent = result;
        if (statusText) statusText.textContent = '음성 인식이 완료되었습니다!';
    };
    
    recognition.onerror = function(event) {
        console.error('음성 인식 오류:', event.error);
        const statusText = document.getElementById('mic_status_text');
        if (statusText) statusText.textContent = '음성 인식 중 오류가 발생했습니다.';
        stopRecording();
    };
    
    recognition.onend = function() {
        isRecording = false;
        const startButton = document.getElementById('start_recording');
        const stopButton = document.getElementById('stop_recording');
        
        if (startButton) startButton.style.display = 'inline-block';
        if (stopButton) stopButton.style.display = 'none';
    };
    
    recognition.start();
}

// 음성 인식 중지
function stopRecording() {
    if (recognition && isRecording) {
        recognition.stop();
        isRecording = false;
    }
}

// 인식된 텍스트 적용
function applyRecognizedText() {
    const recognizedResult = document.getElementById('recognized_result');
    const userInput = document.getElementById('user_input_text');
    
    if (recognizedResult && userInput) {
        const recognizedText = recognizedResult.textContent.trim();
        if (recognizedText) {
            userInput.value = recognizedText;
            hideMicModal();
        }
    }
}

// 질문 데이터 로드
async function loadQuestions() {
    try {
        const params = getParamsFromURL();
        
        let url = `${API_BASE_URL}/questions`;
        let queryParams = new URLSearchParams();
        
        if (params.type === 'aiSettings') {
            // AI 질문 설정
            queryParams.append('use_ai', 'true');
            queryParams.append('difficulty', 'medium'); // 기본값
            queryParams.append('count', params.count.toString());
        } else {
            // 기본 질문 사용 (meme)
            queryParams.append('use_ai', 'false');
        }
        
        const response = await fetch(`${url}?${queryParams}`);
        if (!response.ok) {
            throw new Error('질문 로드 실패');
        }
        const data = await response.json();
        
        // API 응답 구조 확인 및 처리
        if (data.questions && Array.isArray(data.questions)) {
            questions = data.questions;
        } else if (Array.isArray(data)) {
            questions = data;
        } else {
            throw new Error('올바르지 않은 데이터 형식');
        }
        
        console.log('로드된 질문:', questions);
        
        // 질문 로딩 모달 숨기기
        hideQuestionLoadingModal();
        
        // 첫 번째 질문 표시
        showNextQuestion();
        
    } catch (error) {
        console.error('질문 로드 오류:', error);
        // 질문 로딩 모달 숨기기
        hideQuestionLoadingModal();
        alert('질문을 로드하는 중 오류가 발생했습니다.');
    }
}

// 랜덤 질문 선택 (중복 방지)
function getRandomQuestion() {
    if (!questions || questions.length === 0) {
        return "질문을 불러오는 중입니다...";
    }
    
    const availableQuestions = questions.filter((_, index) => !usedQuestions.has(index));
    
    if (availableQuestions.length === 0) {
        // 모든 질문을 사용했다면 사용된 질문 목록 초기화
        usedQuestions.clear();
        return questions[Math.floor(Math.random() * questions.length)];
    }
    
    const randomIndex = Math.floor(Math.random() * availableQuestions.length);
    const selectedQuestion = availableQuestions[randomIndex];
    
    // 원래 인덱스를 찾아서 사용된 질문 목록에 추가
    const originalIndex = questions.indexOf(selectedQuestion);
    usedQuestions.add(originalIndex);
    
    return selectedQuestion;
}

// 다음 질문 표시
function showNextQuestion() {
    if (currentCount >= maxCount) {
        showFinalResult();
        return;
    }
    
    const question = getRandomQuestion();
    const questionElement = document.getElementById('question_text');
    if (questionElement) {
        questionElement.textContent = question;
    }
    
    const inputElement = document.getElementById('user_input_text');
    if (inputElement) {
        inputElement.value = '';
        inputElement.disabled = false;
    }
    
    // 진행 상황 표시 (있다면)
    const progressElement = document.getElementById('progress');
    if (progressElement) {
        progressElement.textContent = `${currentCount + 1} / ${maxCount}`;
    }
    
    // 결과 표시 영역 숨기기
    const resultDiv = document.getElementById('result_display');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
    
    // 버튼 상태 초기화
    const buttonElement = document.getElementById('user_input_button');
    if (buttonElement) {
        buttonElement.disabled = false;
        buttonElement.style.opacity = '1';
    }
}

// 답변 제출
async function submitAnswer() {
    const answer = document.getElementById('user_input_text').value.trim();
    if (!answer) {
        alert('답변을 입력해주세요.');
        return;
    }

    // 버튼 비활성화
    const buttonElement = document.getElementById('user_input_button');
    const inputElement = document.getElementById('user_input_text');
    
    if (buttonElement) {
        buttonElement.disabled = true;
        buttonElement.style.opacity = '0.5';
    }
    if (inputElement) {
        inputElement.disabled = true;
    }

    // 분석 모달 표시
    showAnalysisModal();

    try {
        // API 호출
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: answer })
        });

        if (!response.ok) {
            throw new Error('API 호출 실패');
        }

        const data = await response.json();
        
        // 분석 모달 숨기기
        hideAnalysisModal();
        
        const questionElement = document.getElementById('question_text');
        results.push({
            question: questionElement ? questionElement.textContent : '',
            answer: answer,
            score: data.score
        });

        // 결과 표시
        showResult(data.score);
        
        // 답변 처리 후 answer/answer.html로 이동
        window.location.href = '/answer/answer.html';
        
        // 답변 처리 후 다음 질문으로
        // currentCount++;
        // setTimeout(() => {
        //     showNextQuestion();
        // }, 2000);
        
    } catch (error) {
        console.error('Error:', error);
        // 분석 모달 숨기기
        hideAnalysisModal();
        alert('분석 중 오류가 발생했습니다. 다시 시도해주세요.');
        
        // 오류 시 버튼 다시 활성화
        const buttonElement = document.getElementById('user_input_button');
        const inputElement = document.getElementById('user_input_text');
        
        if (buttonElement) {
            buttonElement.disabled = false;
            buttonElement.style.opacity = '1';
        }
        if (inputElement) {
            inputElement.disabled = false;
        }
    }
}

// 답변 처리 함수 (submitAnswer와 동일하지만 기존 코드 호환성을 위해 유지)
function handleAnswer(answer) {
    const inputElement = document.getElementById('user_input_text');
    if (inputElement) {
        inputElement.value = answer;
    }
    submitAnswer();
}

// 합칠 때 이 부분 부터 지우기
// 결과 표시 함수
function showResult(tfScore) {
    const resultDiv = document.getElementById('result_display') || createResultDisplay();
    
    // T와 F의 비율 계산
    const tPercentage = Math.round((100 - tfScore) / 10) * 10;
    const fPercentage = Math.round(tfScore / 10) * 10;
    
    let resultText = '';
    if (tfScore < 20) resultText = `확실한 T! (T: ${tPercentage}% / F: ${fPercentage}%)`;
    else if (tfScore <= 40) resultText = `T 성향 (T: ${tPercentage}% / F: ${fPercentage}%)`;
    else if (tfScore >= 41 && tfScore <= 59) resultText = `T/F 균형 (T: ${tPercentage}% / F: ${fPercentage}%)`;
    else if (tfScore < 80) resultText = `F 성향 (T: ${tPercentage}% / F: ${fPercentage}%)`;
    else resultText = `확실한 F! (T: ${tPercentage}% / F: ${fPercentage}%)`;

    resultDiv.innerHTML = `
        <div style="margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 10px; text-align: center;">
            <p style="font-size: 1.2em; font-weight: bold; color: #333;">${resultText}</p>
            <div style="width: 100%; height: 30px; background: linear-gradient(to right, #ff4444, #ffff44, #44ff44); border-radius: 15px; position: relative; margin: 1rem 0;">
                <div style="width: 4px; height: 40px; background: black; position: absolute; left: ${tfScore}%; top: -5px;"></div>
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

// 결과 표시 영역 생성
function createResultDisplay() {
    const resultDiv = document.createElement('div');
    resultDiv.id = 'result_display';
    resultDiv.style.display = 'none';
    
    // main 태그 안에 추가
    const main = document.querySelector('main');
    if (main) {
        main.appendChild(resultDiv);
    } else {
        document.body.appendChild(resultDiv);
    }
    
    return resultDiv;
}

// 최종 결과 표시
function showFinalResult() {
    const averageScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
    const tPercentage = Math.round((100 - averageScore) / 10) * 10;
    const fPercentage = Math.round(averageScore / 10) * 10;
    
    let finalText = '';
    if (averageScore < 20) finalText = `당신은 확실한 T입니다! "너 T발C야?"`;
    else if (averageScore <= 40) finalText = `당신은 T 성향이 강합니다.`;
    else if (averageScore >= 41 && averageScore <= 59) finalText = `당신은 T와 F의 균형이 잘 잡혀있습니다.`;
    else if (averageScore < 80) finalText = `당신은 F 성향이 강합니다.`;
    else finalText = `당신은 확실한 F입니다! "너 F구나?"`;
    
    // 전체 페이지를 최종 결과로 교체
    document.body.innerHTML = `
        <div style="display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #f5f5f5;">
            <div style="background: white; padding: 3rem; border-radius: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.1); text-align: center; max-width: 600px;">
                <h1 style="color: #333; margin-bottom: 2rem;">🎯 최종 결과</h1>
                <p style="font-size: 1.5em; font-weight: bold; color: #007bff; margin-bottom: 1rem;">${finalText}</p>
                <p style="font-size: 1.2em; margin-bottom: 2rem;">평균 점수: ${averageScore.toFixed(1)}점 (T: ${tPercentage}% / F: ${fPercentage}%)</p>
                
                <div style="width: 100%; height: 40px; background: linear-gradient(to right, #ff4444, #ffff44, #44ff44); border-radius: 20px; position: relative; margin: 2rem 0;">
                    <div style="width: 4px; height: 50px; background: black; position: absolute; left: ${averageScore}%; top: -5px;"></div>
                </div>
                
                <div style="display: flex; justify-content: space-between; margin: 2rem 0;">
                    <div style="background: rgba(255,68,68,0.1); padding: 1rem; border-radius: 10px; width: 45%;">
                        <strong style="color: #ff4444;">Thinking: ${tPercentage}%</strong>
                    </div>
                    <div style="background: rgba(68,255,68,0.1); padding: 1rem; border-radius: 10px; width: 45%;">
                        <strong style="color: #44ff44;">Feeling: ${fPercentage}%</strong>
                    </div>
                </div>
                
                <button onclick="window.location.href='/'" style="padding: 1rem 2rem; font-size: 1.1em; background: #007bff; color: white; border: none; border-radius: 10px; cursor: pointer; margin-top: 2rem;">
                    다시 시작하기
                </button>
            </div>
        </div>
    `;
}

// 이 주석 위까지 지우면 됨

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    const params = getParamsFromURL();
    maxCount = params.count;
    
    // AI 설정 질문일 때만 로딩 모달 표시
    if (params.type === 'aiSettings') {
        showQuestionLoadingModal();
    }
    
    console.log('질문 개수:', maxCount);
    console.log('질문 타입:', params.type);
    
    // 제출 버튼 클릭 이벤트
    const submitButton = document.getElementById('user_input_button');
    if (submitButton) {
        submitButton.addEventListener('click', submitAnswer);
    }
    
    // Enter 키로도 제출 가능하도록 추가
    const inputText = document.getElementById('user_input_text');
    if (inputText) {
        inputText.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                submitAnswer();
            }
        });
    }

    // 마이크 버튼 클릭 이벤트
    const micButton = document.getElementById('mic_button');
    if (micButton) {
        micButton.addEventListener('click', showMicModal);
    }

    // 마이크 모달 내부 버튼 이벤트
    const startRecordingButton = document.getElementById('start_recording');
    if (startRecordingButton) {
        startRecordingButton.addEventListener('click', startRecording);
    }

    const stopRecordingButton = document.getElementById('stop_recording');
    if (stopRecordingButton) {
        stopRecordingButton.addEventListener('click', stopRecording);
    }

    // 마이크 모달 백그라운드 클릭 시 닫기
    const micModal = document.getElementById('mic_modal');
    if (micModal) {
        micModal.addEventListener('click', function(event) {
            if (event.target === micModal) {
                hideMicModal();
            }
        });
    }
    
    // AI 설정일 때는 지연 후 로드, 밈은 바로 로드
    if (params.type === 'aiSettings') {
        setTimeout(() => {
            loadQuestions();
        }, 100);
    } else {
        loadQuestions();
    }
});