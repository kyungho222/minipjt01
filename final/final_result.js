function showFinalResult(oneLiner, graphHtml, reviewHtml) {
    document.getElementById('questionPage').style.display = 'none';
    document.getElementById('finalPage').style.display = 'block';
    document.getElementById('finalOneLiner').innerHTML = oneLiner;
    document.getElementById('finalGraph').innerHTML = graphHtml;
    document.getElementById('finalReview').innerHTML = reviewHtml;
} 