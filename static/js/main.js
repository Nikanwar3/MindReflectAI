// ===== STEP NAVIGATION =====
let currentStep = 1;

function nextStep(step) {
    // Validate current step before moving
    if (currentStep === 1 && !validateQuestionnaire()) {
        return;
    }

    document.getElementById(`step${currentStep}`).classList.remove('active');
    document.getElementById(`step${step}`).classList.add('active');

    // Update step indicators
    document.querySelectorAll('.step').forEach(s => {
        const sStep = parseInt(s.dataset.step);
        s.classList.remove('active', 'completed');
        if (sStep === step) s.classList.add('active');
        else if (sStep < step) s.classList.add('completed');
    });

    currentStep = step;
    window.scrollTo({ top: document.getElementById('assessment').offsetTop - 80, behavior: 'smooth' });
}

function prevStep(step) {
    document.getElementById(`step${currentStep}`).classList.remove('active');
    document.getElementById(`step${step}`).classList.add('active');

    document.querySelectorAll('.step').forEach(s => {
        const sStep = parseInt(s.dataset.step);
        s.classList.remove('active', 'completed');
        if (sStep === step) s.classList.add('active');
        else if (sStep < step) s.classList.add('completed');
    });

    currentStep = step;
    window.scrollTo({ top: document.getElementById('assessment').offsetTop - 80, behavior: 'smooth' });
}

function validateQuestionnaire() {
    for (let i = 1; i <= 9; i++) {
        const selected = document.querySelector(`input[name="q${i}"]:checked`);
        if (!selected) {
            const question = document.getElementById(`question${i}`);
            question.style.borderColor = '#E17055';
            question.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setTimeout(() => { question.style.borderColor = ''; }, 3000);
            alert(`Please answer Question ${i} before proceeding.`);
            return false;
        }
    }
    return true;
}

// ===== CHARACTER COUNT =====
const userText = document.getElementById('userText');
const charCount = document.getElementById('charCount');

if (userText && charCount) {
    userText.addEventListener('input', () => {
        charCount.textContent = userText.value.length;
    });
}

// ===== FILE UPLOADS =====
const audioFile = document.getElementById('audioFile');
const imageFile = document.getElementById('imageFile');

if (audioFile) {
    audioFile.addEventListener('change', (e) => {
        const name = e.target.files[0]?.name || '';
        document.getElementById('audioFileName').textContent = name ? `Selected: ${name}` : '';
    });
}

if (imageFile) {
    imageFile.addEventListener('change', (e) => {
        const file = e.target.files[0];
        document.getElementById('imageFileName').textContent = file ? `Selected: ${file.name}` : '';

        if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                document.getElementById('imagePreview').innerHTML =
                    `<img src="${ev.target.result}" alt="Preview">`;
            };
            reader.readAsDataURL(file);
        }
    });
}

// Drag & drop
['audioUploadArea', 'imageUploadArea'].forEach(id => {
    const area = document.getElementById(id);
    if (!area) return;

    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('dragover');
    });
    area.addEventListener('dragleave', () => area.classList.remove('dragover'));
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');
        const input = area.querySelector('input[type="file"]');
        if (e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        }
    });
});

// ===== FORM SUBMISSION =====
const form = document.getElementById('assessmentForm');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');

if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading
        loadingOverlay.classList.add('active');

        const formData = new FormData(form);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                loadingOverlay.classList.remove('active');
                return;
            }

            displayResults(data);
        } catch (err) {
            alert('Something went wrong. Please try again.');
            console.error(err);
        }

        loadingOverlay.classList.remove('active');
    });
}

// ===== DISPLAY RESULTS =====
function displayResults(data) {
    // Hide form, show results
    form.style.display = 'none';
    resultsSection.style.display = 'block';

    // Risk Level
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = data.overall_risk;
    riskBadge.className = 'risk-badge ' + data.overall_risk.toLowerCase();

    // Scores
    animateScore('depScore', data.questionnaire.depression_score);
    animateScore('anxScore', data.questionnaire.anxiety_score);
    animateScore('stressScore', data.questionnaire.stress_score);

    // Text Analysis
    const textResults = document.getElementById('textResults');
    const ta = data.text_analysis;
    textResults.innerHTML = `
        <div class="analysis-item">
            <div class="label">Sentiment</div>
            <div class="value">${ta.sentiment_label}</div>
        </div>
        <div class="analysis-item">
            <div class="label">Sentiment Score</div>
            <div class="value">${ta.sentiment_score} / 100</div>
        </div>
        <div class="analysis-item">
            <div class="label">Polarity</div>
            <div class="value">${ta.polarity}</div>
        </div>
        <div class="analysis-item">
            <div class="label">Risk from Text</div>
            <div class="value">${ta.risk_from_text}</div>
        </div>
    `;

    // Audio Features
    const audioResults = document.getElementById('audioResults');
    const af = data.audio_features;
    audioResults.innerHTML = `
        <div class="analysis-item">
            <div class="label">MFCC Mean</div>
            <div class="value">${typeof af.MFCC_Mean === 'number' ? af.MFCC_Mean.toFixed(2) : af.MFCC_Mean}</div>
        </div>
        <div class="analysis-item">
            <div class="label">MFCC Variance</div>
            <div class="value">${typeof af.MFCC_Variance === 'number' ? af.MFCC_Variance.toFixed(2) : af.MFCC_Variance}</div>
        </div>
        <div class="analysis-item">
            <div class="label">Pitch Mean</div>
            <div class="value">${typeof af.Pitch_Mean === 'number' ? af.Pitch_Mean.toFixed(2) : af.Pitch_Mean} Hz</div>
        </div>
        <div class="analysis-item">
            <div class="label">Speech Rate</div>
            <div class="value">${typeof af.Speech_Rate === 'number' ? af.Speech_Rate.toFixed(2) : af.Speech_Rate}</div>
        </div>
    `;

    // Image Features
    const imageResults = document.getElementById('imageResults');
    const imf = data.image_features;
    imageResults.innerHTML = `
        <div class="analysis-item">
            <div class="label">Facial Emotion Variance</div>
            <div class="value">${imf.Facial_Emotion_Variance}</div>
        </div>
        <div class="analysis-item">
            <div class="label">Eye Blink Rate</div>
            <div class="value">${imf.Eye_Blink_Rate}</div>
        </div>
        <div class="analysis-item">
            <div class="label">Smile Intensity</div>
            <div class="value">${imf.Smile_Intensity}</div>
        </div>
        <div class="analysis-item">
            <div class="label">Head Motion Index</div>
            <div class="value">${imf.Head_Motion_Index}</div>
        </div>
    `;

    // ML Prediction
    const mlCard = document.getElementById('mlCard');
    const mlResults = document.getElementById('mlResults');
    if (data.ml_prediction.model_loaded) {
        mlCard.style.display = 'block';
        mlResults.innerHTML = `
            <div class="analysis-item">
                <div class="label">Prediction</div>
                <div class="value">${data.ml_prediction.prediction === 1 ? 'Depressed' : 'Not Depressed'}</div>
            </div>
            <div class="analysis-item">
                <div class="label">Confidence</div>
                <div class="value">${(data.ml_prediction.probability * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    // Recommendations
    const recList = document.getElementById('recommendationsList');
    recList.innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function animateScore(elementId, target) {
    const el = document.getElementById(elementId);
    let current = 0;
    const increment = Math.max(1, Math.floor(target / 20));
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        el.textContent = current;
    }, 40);
}

// ===== RESET =====
function resetForm() {
    form.reset();
    form.style.display = 'block';
    resultsSection.style.display = 'none';
    document.getElementById('audioFileName').textContent = '';
    document.getElementById('imageFileName').textContent = '';
    document.getElementById('imagePreview').innerHTML = '';
    if (charCount) charCount.textContent = '0';

    // Reset steps
    currentStep = 1;
    document.querySelectorAll('.form-step').forEach(s => s.classList.remove('active'));
    document.getElementById('step1').classList.add('active');
    document.querySelectorAll('.step').forEach(s => {
        s.classList.remove('active', 'completed');
    });
    document.querySelector('.step[data-step="1"]').classList.add('active');

    document.getElementById('assessment').scrollIntoView({ behavior: 'smooth' });
}

// ===== SMOOTH SCROLL FOR NAV =====
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
