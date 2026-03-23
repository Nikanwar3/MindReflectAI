import os
import numpy as np
import librosa
import cv2
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from textblob import TextBlob

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ML model and scaler if available
MODEL_LOADED = False
model = None
scaler = None

model_path = os.path.join(os.path.dirname(__file__), "depression_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    MODEL_LOADED = True


# ---- Processing Functions ----

def process_questionnaire(responses):
    depression_score = sum(responses[0:3]) * 4
    anxiety_score = sum(responses[3:6]) * 4
    stress_score = sum(responses[6:9]) * 4
    return depression_score, anxiety_score, stress_score


def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc))
        mfcc_variance = float(np.var(mfcc))

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 200.0

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        speech_rate = len(onset_frames) / duration if duration > 0 else 3.5

        return {
            "MFCC_Mean": mfcc_mean,
            "MFCC_Variance": mfcc_variance,
            "Pitch_Mean": pitch_mean,
            "Speech_Rate": speech_rate
        }
    except Exception:
        return {
            "MFCC_Mean": 0,
            "MFCC_Variance": 5,
            "Pitch_Mean": 200,
            "Speech_Rate": 3.5
        }


def extract_image_features(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return default_image_features()

        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        smiles = smile_cascade.detectMultiScale(face, 1.7, 20)
        smile_intensity = len(smiles)
        facial_variance = float(np.var(face))
        blink_rate = float(np.mean(face))
        head_motion = float(np.std(face))

        return {
            "Facial_Emotion_Variance": facial_variance,
            "Eye_Blink_Rate": blink_rate,
            "Smile_Intensity": smile_intensity,
            "Head_Motion_Index": head_motion
        }
    except Exception:
        return default_image_features()


def default_image_features():
    return {
        "Facial_Emotion_Variance": 0.4,
        "Eye_Blink_Rate": 12,
        "Smile_Intensity": 0.3,
        "Head_Motion_Index": 0.4
    }


def analyze_text_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity      # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1

    negative_words = [
        "sad", "hopeless", "tired", "anxious", "stressed", "depressed",
        "lonely", "worthless", "empty", "numb", "exhausted", "overwhelmed",
        "afraid", "angry", "frustrated", "helpless", "miserable", "pain",
        "suffering", "crying", "lost", "broken", "failure", "hate", "worried"
    ]
    positive_words = [
        "happy", "good", "great", "fine", "okay", "better", "hopeful",
        "excited", "grateful", "calm", "peaceful", "motivated", "confident",
        "strong", "loved", "blessed", "joyful", "energetic", "content"
    ]

    text_lower = text.lower()
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)

    # Sentiment score: 0 (very negative) to 100 (very positive)
    sentiment_score = (polarity + 1) * 50

    # Adjust based on keyword analysis
    keyword_adjustment = (pos_count - neg_count) * 5
    sentiment_score = max(0, min(100, sentiment_score + keyword_adjustment))

    if sentiment_score < 30:
        sentiment_label = "Negative"
        risk_from_text = "High"
    elif sentiment_score < 50:
        sentiment_label = "Somewhat Negative"
        risk_from_text = "Moderate"
    elif sentiment_score < 70:
        sentiment_label = "Neutral"
        risk_from_text = "Low"
    else:
        sentiment_label = "Positive"
        risk_from_text = "Very Low"

    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "sentiment_score": round(sentiment_score, 1),
        "sentiment_label": sentiment_label,
        "risk_from_text": risk_from_text,
        "negative_keywords_found": neg_count,
        "positive_keywords_found": pos_count
    }


def clip(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def stabilize_features(features):
    features["Depression_Score"] = clip(features["Depression_Score"], 0, 40)
    features["Anxiety_Score"] = clip(features["Anxiety_Score"], 0, 40)
    features["Stress_Score"] = clip(features["Stress_Score"], 0, 40)
    features["MFCC_Mean"] = clip(features["MFCC_Mean"], -40, 40)
    features["Pitch_Mean"] = clip(features["Pitch_Mean"], 80, 300)
    features["Speech_Rate"] = clip(features["Speech_Rate"], 1, 7)
    return features


def create_feature_vector(features):
    vector = np.array([[
        features["Depression_Score"],
        features["Anxiety_Score"],
        features["Stress_Score"],
        features.get("Sleep_Quality", 3),
        features.get("Social_Engagement", 3),
        features.get("Daily_App_Usage_Min", 120),
        features.get("Typing_Speed_WPM", 45),
        features.get("Session_Frequency", 10),
        features.get("Idle_Time_Min", 100),
        features["Facial_Emotion_Variance"],
        features["Eye_Blink_Rate"],
        features["Smile_Intensity"],
        features["Head_Motion_Index"],
        features["MFCC_Mean"],
        features["MFCC_Variance"],
        features["Pitch_Mean"],
        features["Speech_Rate"],
        features.get("Heart_Rate_BPM", 80),
        features.get("HRV_Index", 60),
        features.get("Skin_Temperature", 35),
        features.get("GSR_Level", 2)
    ]])
    return vector


def get_risk_level(dep_score, anx_score, stress_score, text_sentiment, probability=None):
    total = dep_score + anx_score + stress_score

    if probability is not None:
        if probability > 0.75:
            ml_risk = "High"
        elif probability > 0.5:
            ml_risk = "Moderate"
        else:
            ml_risk = "Low"
    else:
        ml_risk = None

    if total >= 80:
        score_risk = "High"
    elif total >= 50:
        score_risk = "Moderate"
    else:
        score_risk = "Low"

    # Combine all signals
    risk_signals = [score_risk, text_sentiment.get("risk_from_text", "Low")]
    if ml_risk:
        risk_signals.append(ml_risk)

    if "High" in risk_signals:
        overall = "High"
    elif risk_signals.count("Moderate") >= 2:
        overall = "High"
    elif "Moderate" in risk_signals:
        overall = "Moderate"
    else:
        overall = "Low"

    return overall, score_risk


def get_recommendations(risk_level, dep_score, anx_score, stress_score):
    recs = []

    if risk_level == "High":
        recs.append("We strongly recommend consulting a mental health professional.")
        recs.append("Consider reaching out to a trusted friend or family member.")
        recs.append("If you're in crisis, please contact a helpline immediately.")
    elif risk_level == "Moderate":
        recs.append("Consider speaking with a counselor or therapist.")
        recs.append("Practice regular self-care activities.")
    else:
        recs.append("Continue maintaining your current well-being practices.")

    if dep_score > 20:
        recs.append("Try to engage in activities you previously enjoyed, even in small amounts.")
        recs.append("Maintain a regular sleep schedule and aim for 7-9 hours of sleep.")
    if anx_score > 20:
        recs.append("Practice deep breathing exercises or meditation daily.")
        recs.append("Limit caffeine intake and try progressive muscle relaxation.")
    if stress_score > 20:
        recs.append("Break tasks into smaller, manageable steps.")
        recs.append("Take regular breaks and practice mindfulness.")

    return recs


# ---- Routes ----

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. Questionnaire
        responses = []
        for i in range(1, 10):
            val = request.form.get(f'q{i}', '0')
            responses.append(int(val))

        dep_score, anx_score, stress_score = process_questionnaire(responses)

        # 2. Text sentiment
        user_text = request.form.get('user_text', '')
        text_sentiment = analyze_text_sentiment(user_text) if user_text.strip() else {
            "polarity": 0, "subjectivity": 0, "sentiment_score": 50,
            "sentiment_label": "Neutral", "risk_from_text": "Low",
            "negative_keywords_found": 0, "positive_keywords_found": 0
        }

        # 3. Audio features
        audio_features = None
        audio_file = request.files.get('audio_file')
        audio_path = None
        if audio_file and audio_file.filename:
            filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(audio_path)
            audio_features = extract_audio_features(audio_path)

        if audio_features is None:
            audio_features = {
                "MFCC_Mean": 0, "MFCC_Variance": 5,
                "Pitch_Mean": 200, "Speech_Rate": 3.5
            }

        # 4. Image features
        image_features = None
        image_file = request.files.get('image_file')
        image_path = None
        if image_file and image_file.filename:
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            image_features = extract_image_features(image_path)

        if image_features is None:
            image_features = default_image_features()

        # 5. ML prediction
        features = {
            "Depression_Score": dep_score,
            "Anxiety_Score": anx_score,
            "Stress_Score": stress_score,
            **audio_features,
            **image_features
        }
        features = stabilize_features(features)
        vector = create_feature_vector(features)

        probability = None
        prediction = None
        if MODEL_LOADED:
            vector_scaled = scaler.transform(vector)
            probability = float(model.predict_proba(vector_scaled)[0][1])
            prediction = 1 if probability > 0.6 else 0

        # 6. Risk assessment
        overall_risk, score_risk = get_risk_level(
            dep_score, anx_score, stress_score, text_sentiment, probability
        )

        recommendations = get_recommendations(overall_risk, dep_score, anx_score, stress_score)

        # Cleanup uploaded files
        for path in [audio_path, image_path]:
            if path and os.path.exists(path):
                os.remove(path)

        result = {
            "questionnaire": {
                "depression_score": dep_score,
                "anxiety_score": anx_score,
                "stress_score": stress_score,
                "total_score": dep_score + anx_score + stress_score
            },
            "text_analysis": text_sentiment,
            "audio_features": audio_features,
            "image_features": {
                "Facial_Emotion_Variance": round(image_features["Facial_Emotion_Variance"], 2),
                "Eye_Blink_Rate": round(image_features["Eye_Blink_Rate"], 2),
                "Smile_Intensity": round(image_features["Smile_Intensity"], 2),
                "Head_Motion_Index": round(image_features["Head_Motion_Index"], 2)
            },
            "ml_prediction": {
                "model_loaded": MODEL_LOADED,
                "prediction": prediction,
                "probability": round(probability, 4) if probability is not None else None
            },
            "overall_risk": overall_risk,
            "recommendations": recommendations
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
