// 예시 데이터 (실제 데이터로 교체 가능)
const question = "“너 T야?”";
const barPercent = 80; // 1~100
const characterImg = "../images/Simple_angry.png";

// 적용
document.getElementById('questionText').innerText = question;
document.getElementById('barFill').style.width = barPercent + "%";
document.getElementById('barValue').innerText = barPercent;
document.getElementById('characterImg').src = characterImg;

// 버튼 이벤트 예시
document.getElementById('reviewBtn').onclick = async function() {
  // 기존 예시 데이터 제거
  document.getElementById('modalQuestion').innerText = '왜 이렇게 말수가 줄었어?';
  document.getElementById('modalAnswer').innerText = '요즘 생각이 많아져서 그래';
  document.getElementById('modalDetail').innerText = '너의 대답은 극도의 T야';
  document.getElementById('modalTip').style.display = 'none';
  document.getElementById('modalAlternative').style.display = 'none';
  document.getElementById('reviewDetailModal').style.display = 'flex';
  // 캐릭터는 계속 보이게 (숨기지 않음)

  // 실제 API 호출 예시 (질문/답변/점수는 실제 값으로 대체 필요)
  try {
    const response = await fetch('/detailed_analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: document.getElementById('questionText').innerText,
        answer: document.getElementById('modalAnswer').innerText,
        score: parseFloat(document.getElementById('barValue').innerText)
      })
    });
    if (response.ok) {
      const data = await response.json();
      // 실천팁/대안답변 분리 표시
      document.getElementById('modalTip').innerText = data.tip;
      document.getElementById('modalTip').style.display = '';
      document.getElementById('modalAlternative').innerText = data.alternative;
      document.getElementById('modalAlternative').style.display = '';
    }
  } catch (e) {
    // 오류 시 무시 (예시 데이터만 표시)
  }
};
document.getElementById('nextBtn').onclick = function() {
  alert('다음 질문으로 이동');
  // location.href = 'next.html'; // 실제 다음 페이지로 이동
};
document.getElementById('modalCloseBtn').onclick = function() {
  document.getElementById('reviewDetailModal').style.display = 'none';
  document.getElementById('characterImg').style.display = '';
}; 