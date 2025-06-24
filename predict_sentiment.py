import joblib
import re
import string
import random
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from colorama import Fore, Style, init

init(autoreset=True)

model_path = "model/sentiment_model.joblib"
vectorizer_path = "model/tfidf_vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

negative_keywords = [
    "inflation", "crash", "decline", "recession", "unemployment",
    "interest rate", "devaluation", "trade war", "debt crisis", "bear market",
    "hate", "sucks", "worst", "bad", "fail", "loss"
]

positive_tips = [
    "Great! Keep up the good energy!",
    "Stay focused and maintain this positive outlook!",
    "Keep pushing forward, you're doing well!",
    "Use this positivity to inspire others!"
]

negative_tips = [
    "Take a deep breath and stay hopeful.",
    "You're stronger than your struggles.",
    "This too shall pass. Stay resilient.",
    "Focus on what you can control."
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def extract_keywords(text, top_n=3):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([w for w, _ in sorted_words[:top_n]]) if sorted_words else "No keywords"

def contains_negative_keywords(text):
    return any(kw in text.lower() for kw in negative_keywords)

def adjust_sentiment(text, label, confidence):
    if contains_negative_keywords(text):
        return f"{Fore.RED}Negative 😞 (rule override){Style.RESET_ALL}", "negative"
    if "Positive" in label and confidence < 65:
        return f"{Fore.YELLOW}Neutral/Uncertain 🤔 ({confidence:.1f}% confidence){Style.RESET_ALL}", "neutral"
    return label, "positive" if "Positive" in label else "negative"

def flair_confidence_label(label, confidence):
    if "Negative" in label:
        return label
    if "Neutral" in label:
        return Fore.YELLOW + label + Style.RESET_ALL + " ⚠️"
    if confidence > 90:
        return Fore.GREEN + label + Style.RESET_ALL + " 🚀"
    elif confidence < 60:
        return Fore.YELLOW + label + Style.RESET_ALL + " ⚠️"
    else:
        return label

def get_motivation(sentiment):
    if sentiment == "positive":
        return random.choice(positive_tips)
    elif sentiment == "negative":
        return random.choice(negative_tips)
    else:
        return "Keep reflecting and learning."

if __name__ == "__main__":
    print(Fore.CYAN + "📢 Welcome to the Sentiment Prediction Tool!")
    print("Type 'q' to quit or 'history' to view last 5 entries." + Style.RESET_ALL)
    history = []

    while True:
        try:
            user_input = input("\nEnter your text: ").strip()
            if not user_input:
                print(Fore.RED + "⚠️ Please enter something." + Style.RESET_ALL)
                continue

            if user_input.lower() == 'q':
                print(Fore.GREEN + "👋 Goodbye! Stay motivated!" + Style.RESET_ALL)
                break

            if user_input.lower() == 'history':
                if not history:
                    print(Fore.YELLOW + "No entries yet." + Style.RESET_ALL)
                else:
                    print(Fore.MAGENTA + "📜 Last 5 entries:" + Style.RESET_ALL)
                    for i, (text, label) in enumerate(history[-5:], 1):
                        print(f"{i}. \"{text}\" => {label}")
                continue

            cleaned = clean_text(user_input)
            if not cleaned:
                print(Fore.RED + "⚠️ Your input was not valid after cleaning." + Style.RESET_ALL)
                continue

            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            confidence = max(proba) * 100

            label = f"Positive 😊 ({confidence:.1f}% confidence)" if prediction == 1 else f"Negative 😞 ({confidence:.1f}% confidence)"
            adjusted_label, sentiment_type = adjust_sentiment(user_input, label, confidence)
            final_label = flair_confidence_label(adjusted_label, confidence)
            motivation = get_motivation(sentiment_type)
            keywords = extract_keywords(user_input)

            history.append((user_input, final_label))

            print(Fore.MAGENTA + "👉 Prediction: " + Style.RESET_ALL + final_label)
            print(Fore.YELLOW + f"🔑 Keywords: {keywords}" + Style.RESET_ALL)
            print(Fore.GREEN + f"💬 Tip: {motivation}" + Style.RESET_ALL)
            print("-" * 40)

        except Exception as e:
            print(Fore.RED + f"❌ Error occurred: {str(e)}" + Style.RESET_ALL)