import os
# Configure local directory for Keras to avoid permission errors
os.environ.setdefault("KERAS_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keras"))

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import requests
import os
import math
from werkzeug.utils import secure_filename  
import uuid
from predict_disease import predict_disease
from utils.disease_solutions import get_disease_solution
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from utils.translator import translate_text
import google.generativeai as genai
import feedparser

# IMPORTANT: Replace these placeholders with your actual API keys or environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
DATAGOV_API_KEY = os.environ.get("DATAGOV_API_KEY", "YOUR_DATAGOV_API_KEY")

# Configure Gemini
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)


def translate_global(text):
    lang = session.get("lang", "en")

    if lang == "en":
        return text

    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text

df_chatbot = pd.read_csv("data/agriculture_chatbot_500.csv")
df_markets_loc = pd.read_csv("data/markets_location.csv")

chat_questions = df_chatbot["question"].tolist()
chat_answers = df_chatbot["answer"].tolist()

try:
    chat_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception:
    chat_model = SentenceTransformer('all-MiniLM-L6-v2')
chat_embeddings = chat_model.encode(chat_questions)

SUPPORTED_LANGUAGES = ["en", "ml", "ta", "te", "kn", "hi"]

def auto_translate(data, lang):
    if lang == "en":
        return data

    if isinstance(data, dict):
        return {k: auto_translate(v, lang) for k, v in data.items()}

    if isinstance(data, list):
        return [auto_translate(i, lang) for i in data]

    if isinstance(data, str):
        try:
            return GoogleTranslator(source='auto', target=lang).translate(data)
        except:
            return data

    return data

# ---------------- LANGUAGE DICTIONARY ----------------
LANGUAGES = {

    "en": {
        "login": "Login",
        "register": "Register",
        "email": "Email",
        "phone": "Phone",
        "name": "Name",
        "state": "State",
        "dashboard": "Dashboard",
        "subtitle": "Your Smart Farming Companion",

        "disease": "Disease Detector",
        "disease_desc": "Scan crops or upload images to detect diseases.",
        "disease_btn": "Scan / Upload",

        "schemes": "Government Schemes",
        "schemes_desc": "Central & State agriculture schemes.",
        "schemes_btn": "View Schemes",

        "market": "Market Near Me",
        "market_desc": "Find nearby markets to sell crops.",
        "market_btn": "Find Markets",

        "prices": "Market Prices",
        "prices_desc": "Check current crop prices.",
        "prices_btn": "View Prices",

        "search_crop": "Search crop...",
        "crop_not_found": "❌ Crop not found",
        "price": "Price",

        "weather": "Weather Forecast",
        "weather_desc": "Get live weather updates.",
        "weather_btn": "Enable Location",

        "soil": "Soil Information",
        "soil_desc": "Explore soil advantages and crop suitability.",
        "soil_btn": "View Soil Info",

"calendar": "Farmer Calendar",
"select_state": "-- Select State --",
"loading_calendar": "Loading seasonal plan...",
"sowing": "Sowing",
"planting": "Planting",
"harvest": "Harvest",
"error_loading_calendar": "Error loading calendar",
"calendar_btn": "Open Calendar",

        "ai": "My Farmer (AI)",
        "ai_desc": "Friendly AI for agriculture guidance.",
        "ai_btn": "Chat Now",

        "schemes": "Government Schemes",
"central_schemes": "Central Government Schemes",
"select_state": "Select Your State",
"official_site": "Official Website",

"pmkisan_title": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
"pmkisan_desc": "Provides ₹6000 per year income support to farmers.",

"pmfby_title": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
"pmfby_desc": "Insurance coverage for crop losses due to natural disasters.",

"soilhealth_title": "Soil Health Card Scheme",
"soilhealth_desc": "Provides soil testing and fertility recommendations.",

"weather_dashboard": "Weather Dashboard",
"enter_place": "Enter city / place",
"add_place": "Add Place",
"my_location": "My Location",
"latest_weather_news": "Latest Weather News",
"temp": "Temp",
"humidity": "Humidity",
"wind": "Wind",
"rain": "Rain",
"condition": "Condition",
"place_not_found": "Place not found",
"enter_valid_place": "Enter a valid place",

"soil": "Soil Information",
"soil_title": "Soil Information System",
"select_soil": "-- Select Soil Type --",
"loading_soil": "Loading soil insights...",
"advantages": "Advantages",
"limitations": "Limitations",
"best_crops": "Best Crops",
"fertilizer_tips": "Fertilizer Tips",
"pest_management": "Pest Management",
"error_loading_soil": "Error loading soil data"
    },

    "ml": {
        "login": "ലോഗിൻ",
        "register": "രജിസ്റ്റർ",
        "email": "ഇമെയിൽ",
        "phone": "ഫോൺ",
        "name": "പേര്",
        "state": "സംസ്ഥാനം",
        "dashboard": "ഡാഷ്ബോർഡ്",
        "subtitle": "നിങ്ങളുടെ സ്മാർട്ട് കൃഷി സഹായി",

        "disease": "രോഗം കണ്ടെത്തൽ",
        "disease_desc": "വിളകളിലെ രോഗങ്ങൾ കണ്ടെത്തുക.",
        "disease_btn": "സ്കാൻ ചെയ്യുക",

        "schemes": "സർക്കാർ പദ്ധതികൾ",
        "schemes_desc": "കേന്ദ്ര & സംസ്ഥാന കൃഷി പദ്ധതികൾ.",
        "schemes_btn": "പദ്ധതികൾ കാണുക",

        "market": "അടുത്തുള്ള മാർക്കറ്റ്",
        "market_desc": "വിളകൾ വിൽക്കാൻ മാർക്കറ്റുകൾ കണ്ടെത്തുക.",
        "market_btn": "മാർക്കറ്റുകൾ",

        "prices": "വിപണി വിലകൾ",
        "prices_desc": "ഇന്നത്തെ വിളവിലകൾ പരിശോധിക്കുക.",
        "prices_btn": "വില കാണുക",

        "search_crop": "വിള തിരയുക...",
        "crop_not_found": "❌ വിള കണ്ടെത്തിയില്ല",
        "price": "വില",

        "weather": "കാലാവസ്ഥ",
        "weather_desc": "തത്സമയ കാലാവസ്ഥ വിവരം.",
        "weather_btn": "ലൊക്കേഷൻ",

        "soil": "മണ്ണ് വിവരം",
        "soil_desc": "മണ്ണിന്റെ ഗുണങ്ങളും വിളകളും പരിശോധിക്കുക.",
        "soil_btn": "മണ്ണ് കാണുക",

        "calendar": "കർഷക കലണ്ടർ",
"select_state": "-- സംസ്ഥാനം തിരഞ്ഞെടുക്കുക --",
"loading_calendar": "സീസൺ പദ്ധതി ലോഡിംഗ്...",
"sowing": "വിത്തിടൽ",
"planting": "നട്ട് വളർത്തൽ",
"harvest": "വിളവെടുപ്പ്",
"error_loading_calendar": "കലണ്ടർ ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല",
"calendar_btn": "കലണ്ടർ തുറക്കുക",

        "ai": "എന്റെ ഫാർമർ AI",
        "ai_desc": "കൃഷിക്ക് സൗഹൃദ AI സഹായി.",
        "ai_btn": "ചാറ്റ്",

        "schemes": "സർക്കാർ പദ്ധതികൾ",
"central_schemes": "കേന്ദ്ര സർക്കാർ പദ്ധതികൾ",
"select_state": "നിങ്ങളുടെ സംസ്ഥാനം തിരഞ്ഞെടുക്കുക",
"official_site": "അധികൃത വെബ്സൈറ്റ്",

"pmkisan_title": "പ്രധാനമന്ത്രി കിസാൻ സമ്മാൻ നിധി (PM-KISAN)",
"pmkisan_desc": "കർഷകർക്ക് പ്രതിവർഷം ₹6,000 നേരിട്ടുള്ള വരുമാന സഹായം.",

"pmfby_title": "പ്രധാനമന്ത്രി ഫസൽ ബിമാ യോജന",
"pmfby_desc": "പ്രകൃതിദുരന്തങ്ങളാൽ വിളനഷ്ടം സംഭവിക്കുമ്പോൾ ഇൻഷുറൻസ് സംരക്ഷണം.",

"soilhealth_title": "സോയിൽ ഹെൽത്ത് കാർഡ് പദ്ധതി",
"soilhealth_desc": "മണ്ണ് പരിശോധനയും ഫലഭൂയിഷ്ഠത വിലയിരുത്തലും.",

"weather_dashboard": "കാലാവസ്ഥ ഡാഷ്ബോർഡ്",
"enter_place": "നഗരം / സ്ഥലം നൽകുക",
"add_place": "സ്ഥലം ചേർക്കുക",
"my_location": "എന്റെ ലൊക്കേഷൻ",
"latest_weather_news": "പുതിയ കാലാവസ്ഥ വാർത്തകൾ",
"temp": "താപനില",
"humidity": "ആർദ്രത",
"wind": "കാറ്റ്",
"rain": "മഴ",
"condition": "സ്ഥിതി",
"place_not_found": "സ്ഥലം കണ്ടെത്താനായില്ല",
"enter_valid_place": "ശരിയായ സ്ഥലം നൽകുക",

"soil": "മണ്ണ് വിവരം",
"soil_title": "മണ്ണ് വിവര സംവിധാനം",
"select_soil": "-- മണ്ണ് തരം തിരഞ്ഞെടുക്കുക --",
"loading_soil": "മണ്ണ് വിവരങ്ങൾ ലോഡിംഗ്...",
"advantages": "ഗുണങ്ങൾ",
"limitations": "പരിമിതികൾ",
"best_crops": "മികച്ച വിളകൾ",
"fertilizer_tips": "വള നിർദ്ദേശങ്ങൾ",
"pest_management": "കീട നിയന്ത്രണം",
"error_loading_soil": "മണ്ണ് ഡാറ്റ ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല"


    },

    "ta": {
        "login": "உள்நுழை",
        "register": "பதிவு",
        "email": "மின்னஞ்சல்",
        "phone": "தொலைபேசி",
        "name": "பெயர்",
        "state": "மாநிலம்",
        "dashboard": "டாஷ்போர்டு",
        "subtitle": "உங்கள் ஸ்மார்ட் விவசாய துணை",

        "disease": "நோய் கண்டறிதல்",
        "disease_desc": "பயிர் நோய்களை கண்டறியவும்.",
        "disease_btn": "ஸ்கேன்",

        "schemes": "அரசு திட்டங்கள்",
        "schemes_desc": "மத்திய & மாநில திட்டங்கள்.",
        "schemes_btn": "பார்க்க",

        "market": "அருகிலுள்ள சந்தை",
        "market_desc": "விற்க சந்தைகளை கண்டறியவும்.",
        "market_btn": "சந்தைகள்",

        "prices": "சந்தை விலைகள்",
        "prices_desc": "இன்றைய விலைகள்.",
        "prices_btn": "விலைகள்",

        "search_crop": "பயிர் தேடுக...",
        "crop_not_found": "❌ பயிர் கிடைக்கவில்லை",
        "price": "விலை",

        "weather": "வானிலை",
        "weather_desc": "நேரடி வானிலை தகவல்.",
        "weather_btn": "இடம்",

        "soil": "மண் தகவல்",
        "soil_desc": "மண் நன்மைகள் & பயிர்கள்.",
        "soil_btn": "மண் பார்க்க",
"calendar": "விவசாய காலண்டர்",
"select_state": "-- மாநிலத்தைத் தேர்ந்தெடுக்கவும் --",
"loading_calendar": "சீசன் திட்டம் ஏற்றப்படுகிறது...",
"sowing": "விதைப்பு",
"planting": "நட்டு வளர்த்தல்",
"harvest": "அறுவடை",
"error_loading_calendar": "காலண்டரை ஏற்ற முடியவில்லை",
"calendar_btn": "காலண்டர் திறக்க",
        "ai": "என் விவசாய AI",
        "ai_desc": "விவசாய வழிகாட்டி AI.",
        "ai_btn": "அரட்டை",

        "schemes": "அரசு திட்டங்கள்",
"central_schemes": "மத்திய அரசு திட்டங்கள்",
"select_state": "உங்கள் மாநிலத்தைத் தேர்ந்தெடுக்கவும்",
"official_site": "அதிகாரப்பூர்வ வலைத்தளம்",

"pmkisan_title": "பிரதான் மந்திரி கிசான் சம்மான் நிதி (PM-KISAN)",
"pmkisan_desc": "விவசாயிகளுக்கு வருடத்திற்கு ₹6,000 நேரடி வருமான ஆதரவு.",

"pmfby_title": "பிரதான் மந்திரி பயிர் காப்பீட்டு திட்டம்",
"pmfby_desc": "பயிர் இழப்புகளை பாதுகாக்க காப்பீட்டு திட்டம்.",

"soilhealth_title": "மண் சுகாதார அட்டை திட்டம்",
"soilhealth_desc": "மண் பரிசோதனை மற்றும் உர மதிப்பீடு.",

"weather_dashboard": "வானிலை டாஷ்போர்டு",
"enter_place": "நகரம் / இடம் உள்ளிடவும்",
"add_place": "இடம் சேர்க்க",
"my_location": "என் இருப்பிடம்",
"latest_weather_news": "சமீபத்திய வானிலை செய்திகள்",
"temp": "வெப்பநிலை",
"humidity": "ஈரப்பதம்",
"wind": "காற்று",
"rain": "மழை",
"condition": "நிலை",
"place_not_found": "இடம் கிடைக்கவில்லை",
"enter_valid_place": "சரியான இடம் உள்ளிடவும்",

"soil": "மண் தகவல்",
"soil_title": "மண் தகவல் அமைப்பு",
"select_soil": "-- மண் வகையைத் தேர்ந்தெடுக்கவும் --",
"loading_soil": "மண் தகவல் ஏற்றப்படுகிறது...",
"advantages": "நன்மைகள்",
"limitations": "குறைகள்",
"best_crops": "சிறந்த பயிர்கள்",
"fertilizer_tips": "உரம் ஆலோசனை",
"pest_management": "பூச்சி கட்டுப்பாடு",
"error_loading_soil": "மண் தரவை ஏற்ற முடியவில்லை"
    },

    "te": {
        "login": "లాగిన్",
        "register": "నమోదు",
        "email": "ఈమెయిల్",
        "phone": "ఫోన్",
        "name": "పేరు",
        "state": "రాష్ట్రం",
        "dashboard": "డాష్‌బోర్డ్",
        "subtitle": "మీ స్మార్ట్ వ్యవసాయ సహాయకుడు",

        "disease": "రోగ నిర్ధారణ",
        "disease_desc": "పంటల వ్యాధులను గుర్తించండి.",
        "disease_btn": "స్కాన్",

        "schemes": "ప్రభుత్వ పథకాలు",
        "schemes_desc": "కేంద్ర & రాష్ట్ర పథకాలు.",
        "schemes_btn": "చూడండి",

        "market": "సమీప మార్కెట్",
        "market_desc": "విక్రయానికి మార్కెట్లు.",
        "market_btn": "మార్కెట్లు",

        "prices": "మార్కెట్ ధరలు",
        "prices_desc": "నేటి ధరలు.",
        "prices_btn": "ధరలు",

        "search_crop": "పంట వెతకండి...",
        "crop_not_found": "❌ పంట కనబడలేదు",
        "price": "ధర",

        "weather": "వాతావరణం",
        "weather_desc": "ప్రత్యక్ష వాతావరణ సమాచారం.",
        "weather_btn": "లొకేషన్",

        "soil": "మట్టి సమాచారం",
        "soil_desc": "మట్టి ప్రయోజనాలు & పంటలు.",
        "soil_btn": "మట్టి చూడండి",

        "calendar": "రైతు క్యాలెండర్",
"select_state": "-- రాష్ట్రాన్ని ఎంచుకోండి --",
"loading_calendar": "సీజన్ ప్రణాళిక లోడ్ అవుతోంది...",
"sowing": "విత్తనం",
"planting": "నాటడం",
"harvest": "పంట కోత",
"error_loading_calendar": "క్యాలెండర్ లోడ్ చేయలేకపోయాం",
"calendar_btn": "క్యాలెండర్ తెరవండి",

        "ai": "నా రైతు AI",
        "ai_desc": "వ్యవసాయ మార్గదర్శక AI.",
        "ai_btn": "చాట్",

        "schemes": "ప్రభుత్వ పథకాలు",
        "central_schemes": "కేంద్ర ప్రభుత్వం పథకాలు",
        "select_state": "మీ రాష్ట్రాన్ని ఎంచుకోండి",
        "official_site": "అధికారిక వెబ్‌సైట్",

        "pmkisan_title": "ప్రధాన మంత్రి కిసాన్ సమ్మాన్ నిధి (PM-KISAN)",
        "pmkisan_desc": "రైతులకు సంవత్సరానికి ₹6,000 నేరుగా ఆదాయం సహాయం.",

        "pmfby_title": "ప్రధాన మంత్రి పంట బీమా యోజన",
        "pmfby_desc": "ప్రకృతి విపత్తుల వల్ల పంట నష్టానికి బీమా రక్షణ.",

        "soilhealth_title": "మట్టి ఆరోగ్య కార్డ్ పథకం",
        "soilhealth_desc": "మట్టి పరీక్ష మరియు సారాంశ విశ్లేషణ.",

        "weather_dashboard": "వాతావరణ డాష్‌బోర్డ్",
        "enter_place": "నగరం / స్థలం నమోదు చేయండి",
        "add_place": "స్థలం చేర్చండి",
        "my_location": "నా లొకేషన్",
        "latest_weather_news": "తాజా వాతావరణ వార్తలు",
        "temp": "ఉష్ణోగ్రత",
        "humidity": "ఆర్ద్రత",
        "wind": "గాలి",
        "rain": "వర్షం",
        "condition": "స్థితి",
        "place_not_found": "స్థలం కనబడలేదు",
        "enter_valid_place": "సరైన స్థలం నమోదు చేయండి",

        "soil": "మట్టి సమాచారం",
        "soil_title": "మట్టి సమాచార వ్యవస్థ",
        "select_soil": "-- మట్టి రకం ఎంచుకోండి --",
        "loading_soil": "మట్టి సమాచారం లోడ్ అవుతోంది...",
        "advantages": "ప్రయోజనాలు",
        "limitations": "పరిమితులు",
        "best_crops": "ఉత్తమ పంటలు",
        "fertilizer_tips": "ఎరువు సూచనలు",
        "pest_management": "పురుగు నియంత్రణ",
        "error_loading_soil": "మట్టి డేటా లోడ్ కాలేదు"
    },

    "kn": {
        "login": "ಲಾಗಿನ್",
        "register": "ನೋಂದಣಿ",
        "email": "ಇಮೇಲ್",
        "phone": "ಫೋನ್",
        "name": "ಹೆಸರು",
        "state": "ರಾಜ್ಯ",
        "dashboard": "ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "subtitle": "ನಿಮ್ಮ ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ",

        "disease": "ರೋಗ ಪತ್ತೆ",
        "disease_desc": "ಬೆಳೆ ರೋಗಗಳನ್ನು ಗುರುತಿಸಿ.",
        "disease_btn": "ಸ್ಕ್ಯಾನ್",

        "schemes": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        "schemes_desc": "ಕೇಂದ್ರ & ರಾಜ್ಯ ಯೋಜನೆಗಳು.",
        "schemes_btn": "ನೋಡಿ",

        "market": "ಹತ್ತಿರದ ಮಾರುಕಟ್ಟೆ",
        "market_desc": "ಮಾರಾಟಕ್ಕೆ ಮಾರುಕಟ್ಟೆಗಳು.",
        "market_btn": "ಮಾರುಕಟ್ಟೆ",

        "prices": "ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು",
        "prices_desc": "ಇಂದಿನ ಬೆಲೆಗಳು.",
        "prices_btn": "ಬೆಲೆ",

        "search_crop": "ಬೆಳೆ ಹುಡುಕಿ...",
        "crop_not_found": "❌ ಬೆಳೆ ಸಿಗಲಿಲ್ಲ",
        "price": "ಬೆಲೆ",

        "weather": "ಹವಾಮಾನ",
        "weather_desc": "ನೇರ ಹವಾಮಾನ ಮಾಹಿತಿ.",
        "weather_btn": "ಲೊಕೇಶನ್",

        "soil": "ಮಣ್ಣು ಮಾಹಿತಿ",
        "soil_desc": "ಮಣ್ಣು ಗುಣಗಳು & ಬೆಳೆಗಳು.",
        "soil_btn": "ಮಣ್ಣು ನೋಡಿ",

        "calendar": "ರೈತ ಕ್ಯಾಲೆಂಡರ್",
"select_state": "-- ರಾಜ್ಯವನ್ನು ಆಯ್ಕೆಮಾಡಿ --",
"loading_calendar": "ಸೀಸನ್ ಯೋಜನೆ ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
"sowing": "ಬಿತ್ತನೆ",
"planting": "ನೆಡುವುದು",
"harvest": "ಕೊಯ್ಲು",
"error_loading_calendar": "ಕ್ಯಾಲೆಂಡರ್ ಲೋಡ್ ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ",
"calendar_btn": "ಕ್ಯಾಲೆಂಡರ್ ತೆರೆಯಿರಿ",
        "ai": "ನನ್ನ ರೈತ AI",
        "ai_desc": "ಕೃಷಿ ಮಾರ್ಗದರ್ಶಕ AI.",
        "ai_btn": "ಚಾಟ್",

        "schemes": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
"central_schemes": "ಕೇಂದ್ರ ಸರ್ಕಾರದ ಯೋಜನೆಗಳು",
"select_state": "ನಿಮ್ಮ ರಾಜ್ಯವನ್ನು ಆಯ್ಕೆಮಾಡಿ",
"official_site": "ಅಧಿಕೃತ ವೆಬ್‌ಸೈಟ್",

"pmkisan_title": "ಪ್ರಧಾನಮಂತ್ರಿ ಕಿಸಾನ್ ಸಮ್ಮಾನ್ ನಿಧಿ (PM-KISAN)",
"pmkisan_desc": "ರೈತರಿಗೆ ಪ್ರತಿವರ್ಷ ₹6,000 ನೇರ ಆದಾಯ ನೆರವು.",

"pmfby_title": "ಪ್ರಧಾನಮಂತ್ರಿ ಬೆಳೆ ವಿಮಾ ಯೋಜನೆ",
"pmfby_desc": "ಪ್ರಕೃತಿ ವಿಪತ್ತುಗಳಿಂದ ಬೆಳೆ ನಷ್ಟಕ್ಕೆ ವಿಮಾ ರಕ್ಷಣೆ.",

"soilhealth_title": "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಯೋಜನೆ",
"soilhealth_desc": "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮತ್ತು ಉರ್ವರತೆ ಮೌಲ್ಯಮಾಪನ.",

"weather_dashboard": "ಹವಾಮಾನ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
"enter_place": "ನಗರ / ಸ್ಥಳ ನಮೂದಿಸಿ",
"add_place": "ಸ್ಥಳ ಸೇರಿಸಿ",
"my_location": "ನನ್ನ ಸ್ಥಳ",
"latest_weather_news": "ಇತ್ತೀಚಿನ ಹವಾಮಾನ ಸುದ್ದಿ",
"temp": "ತಾಪಮಾನ",
"humidity": "ಆದ್ರತೆ",
"wind": "ಗಾಳಿ",
"rain": "ಮಳೆ",
"condition": "ಸ್ಥಿತಿ",
"place_not_found": "ಸ್ಥಳ ಸಿಗಲಿಲ್ಲ",
"enter_valid_place": "ಸರಿಯಾದ ಸ್ಥಳ ನಮೂದಿಸಿ",

"soil": "ಮಣ್ಣು ಮಾಹಿತಿ",
"soil_title": "ಮಣ್ಣು ಮಾಹಿತಿ ವ್ಯವಸ್ಥೆ",
"select_soil": "-- ಮಣ್ಣಿನ ಪ್ರಕಾರ ಆಯ್ಕೆಮಾಡಿ --",
"loading_soil": "ಮಣ್ಣು ಮಾಹಿತಿ ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
"advantages": "ಪ್ರಯೋಜನಗಳು",
"limitations": "ಮಿತಿಗಳು",
"best_crops": "ಉತ್ತಮ ಬೆಳೆಗಳು",
"fertilizer_tips": "ಸಾರ ಸಲಹೆಗಳು",
"pest_management": "ಕೀಟ ನಿಯಂತ್ರಣ",
"error_loading_soil": "ಮಣ್ಣು ಡೇಟಾ ಲೋಡ್ ಆಗಲಿಲ್ಲ"
    },

    "hi": {
        "login": "लॉगिन",
        "register": "पंजीकरण",
        "email": "ईमेल",
        "phone": "फ़ोन",
        "name": "नाम",
        "state": "राज्य",
        "dashboard": "डैशबोर्ड",
        "subtitle": "आपका स्मार्ट कृषि साथी",

        "disease": "रोग पहचान",
        "disease_desc": "फसल रोग पहचानें.",
        "disease_btn": "स्कैन",

        "schemes": "सरकारी योजनाएँ",
        "schemes_desc": "केंद्र & राज्य योजनाएँ.",
        "schemes_btn": "देखें",

        "market": "नज़दीकी बाज़ार",
        "market_desc": "फसल बेचने के बाज़ार.",
        "market_btn": "बाज़ार",

        "prices": "बाज़ार मूल्य",
        "prices_desc": "आज के मूल्य.",
        "prices_btn": "मूल्य",

        "search_crop": "फसल खोजें...",
        "crop_not_found": "❌ फसल नहीं मिली",
        "price": "मूल्य",

        "weather": "मौसम",
        "weather_desc": "लाइव मौसम जानकारी.",
        "weather_btn": "लोकेशन",

        "soil": "मिट्टी जानकारी",
        "soil_desc": "मिट्टी लाभ & उपयुक्त फसलें.",
        "soil_btn": "मिट्टी देखें",

        "calendar": "किसान कैलेंडर",
"select_state": "-- राज्य चुनें --",
"loading_calendar": "सीज़न योजना लोड हो रही है...",
"sowing": "बुवाई",
"planting": "रोपण",
"harvest": "कटाई",
"error_loading_calendar": "कैलेंडर लोड नहीं हो सका",
"calendar_btn": "कैलेंडर खोलें",

        "ai": "मेरा किसान AI",
        "ai_desc": "कृषि मार्गदर्शक AI.",
        "ai_btn": "चैट",

        "schemes": "सरकारी योजनाएँ",
"central_schemes": "केंद्रीय सरकारी योजनाएँ",
"select_state": "अपने राज्य का चयन करें",
"official_site": "आधिकारिक वेबसाइट",

"pmkisan_title": "प्रधान मंत्री किसान सम्मान निधि (PM-KISAN)",
"pmkisan_desc": "किसानों के लिए प्रति वर्ष ₹6,000 की आय सहायता।",

"pmfby_title": "प्रधान मंत्री फसल बीमा योजना",
"pmfby_desc": "प्राकृतिक आपदाओं से फसल नुकसान पर बीमा सुरक्षा।",

"soilhealth_title": "मृदा स्वास्थ्य कार्ड योजना",
"soilhealth_desc": "मृदा परीक्षण और उर्वरता मूल्यांकन।",

"weather_dashboard": "मौसम डैशबोर्ड",
"enter_place": "शहर / स्थान दर्ज करें",
"add_place": "स्थान जोड़ें",
"my_location": "मेरी लोकेशन",
"latest_weather_news": "ताज़ा मौसम समाचार",
"temp": "तापमान",
"humidity": "नमी",
"wind": "हवा",
"rain": "बारिश",
"condition": "स्थिति",
"place_not_found": "स्थान नहीं मिला",
"enter_valid_place": "कृपया सही स्थान दर्ज करें",

"soil": "मिट्टी जानकारी",
"soil_title": "मिट्टी जानकारी प्रणाली",
"select_soil": "-- मिट्टी का प्रकार चुनें --",
"loading_soil": "मिट्टी जानकारी लोड हो रही है...",
"advantages": "लाभ",
"limitations": "सीमाएँ",
"best_crops": "उपयुक्त फसलें",
"fertilizer_tips": "उर्वरक सुझाव",
"pest_management": "कीट नियंत्रण",
"error_loading_soil": "मिट्टी डेटा लोड नहीं हुआ"
    }
}
        

# ---------------- Flask App ----------------

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ✅ Inject language globally
@app.context_processor
def inject_globals():

    lang = session.get("lang", "en")

    def translate(text):
        if not text:
            return text
        if lang == "en":
            return text
        try:
            return translate_text(text, lang)
        except:
            return text

    return dict(
        text=LANGUAGES.get(lang, LANGUAGES["en"]),
        translate=translate,
        current_lang=lang
    )

# ---------------- Chatbot Logic ----------------

def chatbot_reply(user_text, lang="en"):
    try:
        user_lower = user_text.lower().strip()

        greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
        if user_lower in greetings:
            reply = "Hello 👋 I’m your Farmer AI assistant. Ask me about crops, fertilizers, diseases, irrigation, or pests 🌾"
            if lang != "en":
                reply = GoogleTranslator(source='auto', target=lang).translate(reply)
            return reply

        thanks = ["thank you", "thanks", "thankyou", "thx", "appreciate it"]
        if any(t in user_lower for t in thanks):
            reply = "You’re welcome 😊 Happy to help with your farming questions!"
            if lang != "en":
                reply = GoogleTranslator(source='auto', target=lang).translate(reply)
            return reply

        if lang != "en":
            user_text_en = GoogleTranslator(source='auto', target='en').translate(user_text)
        else:
            user_text_en = user_text

        # Try Gemini API (testing gemini-3.8-flash, gemini-flash-latest, etc.)
        answer_en = None
        if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
            for model_name in ['gemini-3.8-flash', 'gemini-flash-latest', 'gemini-pro-latest']:
                try:
                    model = genai.GenerativeModel(model_name)
                    prompt = f"You are an expert agricultural assistant. Answer this question from a farmer concisely, practically, and helpfully: {user_text_en}"
                    response = model.generate_content(prompt)
                    if response and response.text:
                        answer_en = response.text
                        break
                except Exception as e:
                    print(f"Gemini API Error with {model_name}:", e)

        # Fallback to CSV if Gemini did not return an answer
        if not answer_en:
            user_embedding = chat_model.encode([user_text_en])
            scores = cosine_similarity(user_embedding, chat_embeddings)[0]
            best_idx = scores.argmax()
            answer_en = chat_answers[best_idx]

        if lang != "en":
            final_answer = GoogleTranslator(source='auto', target=lang).translate(answer_en)
        else:
            final_answer = answer_en

        return final_answer

    except Exception as e:
        print("🔥 CHATBOT ERROR:", e)
        return "Sorry, I couldn't process your question."

# ---------------- Database ----------------

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmers.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Farmer(db.Model):
    __tablename__ = "farmer"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    state = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)

with app.app_context():
    db.create_all()

# ---------------- Market Prices ----------------

df_prices = pd.read_csv("data/market_prices.csv")

# ---------------- Weather API ----------------

API_KEY = "546fa3458a4cb31dd7b7b47cb274188b"

# ---------------- Routes ----------------
#-----------------Welcome-----------------#
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/set-language", methods=["POST"])
def set_language():
    lang = request.form.get("lang")
    session["lang"] = lang
    return redirect(request.referrer or url_for("welcome"))

#------------------------Registration-------------------------#
@app.route("/register", methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        state = request.form['state']
        email = request.form['email']
        phone = request.form['phone']

        existing_user = Farmer.query.filter_by(email=email).first()
        if existing_user:
            lang = session.get("lang", "en")
            flash(translate_text("Email already registered!", lang))
            return redirect(url_for('login'))

        farmer = Farmer(name=name, state=state, email=email, phone=phone)
        db.session.add(farmer)
        db.session.commit()

        session['user_email'] = email
        return redirect(url_for('dashboard'))

    return render_template("register.html")

#--------------------Login-------------------#
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        phone = request.form['phone'].strip()

        user = Farmer.query.filter_by(email=email, phone=phone).first()

        if user:
            session['user_email'] = email
            return redirect(url_for('dashboard'))
        else:
            lang = session.get("lang", "en")
            flash(translate_text("Invalid email or phone!", lang))
            return redirect(url_for('login'))

    return render_template('login.html')

#------------------Logout-------------------#
@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('welcome'))

#----------------Dashboard-----------------#
@app.route("/dashboard")
def dashboard():
    if 'user_email' not in session:
        lang = session.get("lang", "en")
        flash(translate_text("Please login first!", lang))
        return redirect(url_for('login'))

    user = Farmer.query.filter_by(email=session['user_email']).first()
    return render_template("dashboard.html", user=user)

# ---------------- Disease Detector ----------------

@app.route("/disease", methods=["GET", "POST"])
def disease():
    lang = session.get("lang", "en")
    if request.method == "POST":
        try:
            if "image" not in request.files:
                return render_template("disease.html", error=translate_text("No image file uploaded. Please select an image.", lang))

            img = request.files["image"]
            if not img or img.filename == "":
                return render_template("disease.html", error=translate_text("No image selected. Please choose a leaf photo to upload.", lang))

            UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            # Safe unique filename handling (handles Unicode/Telugu/mobile names cleanly)
            orig_name = img.filename or "leaf.jpg"
            ext = os.path.splitext(orig_name)[1].lower()
            if not ext or ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                ext = ".jpg"
            clean_base = secure_filename(os.path.splitext(orig_name)[0]) or "leaf"
            filename = f"{clean_base}_{uuid.uuid4().hex[:8]}{ext}"
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            img.save(img_path)

            selected_crop = request.form.get("crop", session.get("selected_crop", "all")).strip()
            session["selected_crop"] = selected_crop

            disease_name, confidence, raw_alternates = predict_disease(
                img_path, selected_crop=selected_crop, return_alternates=True
            )

            # Evaluate prediction outcome
            alternates_display = []
            if "Not_A_Leaf" in disease_name:
                raw_reason = disease_name.split("Not_A_Leaf:")[1].strip() if "Not_A_Leaf:" in disease_name else "Non-leaf object detected."
                disease_display = f"❌ Blocked: {translate_text(raw_reason, lang)}"
                solution_display = translate_text("Please upload a clear, well-focused close-up photo of an actual crop leaf in good lighting.", lang)
                confidence = 0.0
            elif confidence < 35.0:
                disease_display = translate_text("⚠️ Unrecognized or Low-Confidence Leaf Image", lang)
                solution_display = translate_text("The plant condition could not be identified with sufficient certainty. Please take a clearer photo showing the leaf and any symptoms.", lang)
            elif confidence < 50.0 and selected_crop.lower() in ["all", "auto"]:
                disease_display = translate_text("⚠️ Low-Confidence Cross-Crop Detection", lang)
                solution_display = translate_text("Multiple plant species share similar leaf shapes. To get 100% targeted accuracy, please select your specific crop from the dropdown above.", lang)
            else:
                raw_disease = disease_name
                if "___" in disease_name:
                    parts = disease_name.split("___")
                    crop_str = parts[0].replace("_", " ").strip()
                    cond_str = parts[1].replace("_", " ").strip()
                    disease_display = f"{crop_str} — {cond_str}"
                else:
                    disease_display = disease_name.replace("_", " ")

                if lang != "en":
                    try:
                        disease_display = translate_text(disease_display, lang)
                    except Exception:
                        pass

                solution_display = get_disease_solution(raw_disease, lang)

                # Format alternates
                for alt_name, alt_p in raw_alternates:
                    if "___" in alt_name:
                        aparts = alt_name.split("___")
                        acrop = aparts[0].replace("_", " ").strip()
                        acond = aparts[1].replace("_", " ").strip()
                        alt_title = f"{acrop} — {acond}"
                    else:
                        alt_title = alt_name.replace("_", " ")
                    if lang != "en":
                        try:
                            alt_title = translate_text(alt_title, lang)
                        except Exception:
                            pass
                    alternates_display.append({"name": alt_title, "confidence": alt_p, "raw": alt_name})

            return render_template("disease.html",
                                   disease=disease_display,
                                   confidence=confidence,
                                   solution=solution_display,
                                   image=f"uploads/{filename}",
                                   selected_crop=selected_crop,
                                   alternates=alternates_display)

        except Exception as e:
            return render_template("disease.html", error=f"Error processing image: {str(e)}")

    selected_crop = session.get("selected_crop", "all")
    return render_template("disease.html", selected_crop=selected_crop)

# ---------------- Market ----------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.route("/market")
def market():
    return render_template("market.html")

@app.route("/get-nearby-markets")
def get_nearby_markets():
    try:
        user_lat = float(request.args.get("lat"))
        user_lng = float(request.args.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid location"}), 400

    markets_list = []
    for _, row in df_markets_loc.iterrows():
        dist = haversine(user_lat, user_lng, row['latitude'], row['longitude'])
        markets_list.append({
            "market_name": row['market_name'],
            "district": row['district'],
            "state": row['state'],
            "address": row['address'],
            "distance_km": round(dist, 1),
            "lat": row['latitude'],
            "lng": row['longitude']
        })

    # Sort all APMC markets by true distance to the user
    markets_list.sort(key=lambda x: x['distance_km'])
    return jsonify(markets_list[:5])

CITY_ALIASES = {
    'vizag': 'Visakhapatnam', 'visakha': 'Visakhapatnam', 'visakhapatnam': 'Visakhapatnam', 'waltair': 'Visakhapatnam',
    'hyd': 'Hyderabad', 'hyderabad': 'Hyderabad', 'hydeberad': 'Hyderabad', 'hyderbad': 'Hyderabad', 'hydrabad': 'Hyderabad', 'secunderabad': 'Hyderabad',
    'bza': 'Vijayawada', 'vijayawada': 'Vijayawada', 'bezawada': 'Vijayawada',
    'blr': 'Bangalore', 'bengaluru': 'Bangalore', 'bangalore': 'Bangalore',
    'tpg': 'Tadepalligudem', 'tadepalligudem': 'Tadepalligudem',
    'rjhy': 'Rajahmundry', 'rajahmundry': 'Rajahmundry',
    'kdd': 'Kakinada', 'kakinada': 'Kakinada',
    'gnt': 'Guntur', 'guntur': 'Guntur',
    'knl': 'Kurnool', 'kurnool': 'Kurnool',
    'tpt': 'Tirupati', 'tirupati': 'Tirupati',
    'wgl': 'Warangal', 'warangal': 'Warangal',
    'atp': 'Anantapur', 'anantapur': 'Anantapur',
    'nlr': 'Nellore', 'nellore': 'Nellore',
    'ong': 'Ongole', 'ongole': 'Prakasam',
    'elr': 'Eluru', 'eluru': 'Eluru',
    'bvm': 'Bhimavaram', 'bhimavaram': 'Bhimavaram',
    'tnk': 'Tanuku', 'tanuku': 'Tanuku'
}

@app.route("/search-place-markets")
def search_place_markets():
    place = request.args.get("place", "").strip()
    if not place:
        return jsonify([])

    # Refresh from disk to guarantee full 80+ APMC dataset is loaded
    try:
        df_loc = pd.read_csv("data/markets_location.csv")
    except Exception:
        df_loc = df_markets_loc

    p_clean = place.lower().strip()

    # 1. Alias normalization & fuzzy matching for typos (e.g. Hydeberad -> Hyderabad, vizag -> Visakhapatnam)
    normalized_name = CITY_ALIASES.get(p_clean)
    if not normalized_name:
        known_keys = list(CITY_ALIASES.keys()) + [d.lower() for d in df_loc['district'].unique()] + [m.lower() for m in df_loc['market_name'].unique()]
        matches = difflib.get_close_matches(p_clean, known_keys, n=1, cutoff=0.6)
        if matches:
            matched_key = matches[0]
            normalized_name = CITY_ALIASES.get(matched_key, matched_key.title())
        else:
            normalized_name = place

    # 2. Check direct matches in dataset by normalized name or input
    norm_lower = normalized_name.lower()
    direct_rows = df_loc[
        df_loc['district'].str.contains(norm_lower, case=False, na=False) |
        df_loc['market_name'].str.contains(norm_lower, case=False, na=False) |
        df_loc['address'].str.contains(norm_lower, case=False, na=False) |
        df_loc['district'].str.contains(p_clean, case=False, na=False) |
        df_loc['market_name'].str.contains(p_clean, case=False, na=False)
    ]

    # 3. Geocode with Nominatim to get accurate geographic coordinates (lat, lon)
    lat, lon = None, None
    for search_term in [normalized_name, place]:
        try:
            geo_url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(search_term)}&countrycodes=in&limit=1"
            res = requests.get(geo_url, headers={"User-Agent": "FarmerAssistant/1.0"}, timeout=3)
            if res.status_code == 200 and res.json():
                lat = float(res.json()[0]['lat'])
                lon = float(res.json()[0]['lon'])
                break
        except Exception:
            pass

    # If geocoding failed, but we found direct market matches, use the first market's coords
    if (lat is None or lon is None) and not direct_rows.empty:
        lat = float(direct_rows.iloc[0]['latitude'])
        lon = float(direct_rows.iloc[0]['longitude'])

    # 4. If we have coordinates, sort ALL markets by distance
    if lat is not None and lon is not None:
        all_sorted = []
        for _, row in df_loc.iterrows():
            d = haversine(lat, lon, row['latitude'], row['longitude'])
            all_sorted.append({
                "market_name": row['market_name'],
                "district": row['district'],
                "state": row['state'],
                "address": row['address'],
                "distance_km": round(d, 1),
                "lat": row['latitude'],
                "lng": row['longitude']
            })
        all_sorted.sort(key=lambda x: x['distance_km'])

        # Filter: if closest markets are within 60km, only return those within that local region
        local_markets = [m for m in all_sorted if m['distance_km'] <= 60]
        if local_markets:
            return jsonify(local_markets[:6])
        else:
            return jsonify(all_sorted[:6])

    # If geocoding completely failed, return direct matches if any
    if not direct_rows.empty:
        results = []
        for _, row in direct_rows.head(6).iterrows():
            results.append({
                "market_name": row['market_name'],
                "district": row['district'],
                "state": row['state'],
                "address": row['address'],
                "distance_km": "Local",
                "lat": row['latitude'],
                "lng": row['longitude']
            })
        return jsonify(results)

    # If place could not be identified or found, return empty list (never false fallback!)
    return jsonify([])

#---------------prices---------------------

@app.route("/prices")
def prices():
    states = sorted(df_prices['state'].dropna().unique().tolist())
    commodities = sorted(df_prices['commodity'].dropna().unique().tolist())
    return render_template("prices.html", states=states, commodities=commodities)

@app.route("/get-districts")
def get_districts():
    state = request.args.get("state")
    if not state:
        return jsonify([])
    districts = sorted(df_prices[df_prices['state'] == state]['district'].dropna().unique().tolist())
    return jsonify(districts)

@app.route("/get-commodities")
def get_commodities():
    state = request.args.get("state")
    district = request.args.get("district")
    filtered = df_prices
    if state:
        filtered = filtered[filtered["state"] == state]
    if district:
        filtered = filtered[filtered["district"] == district]
    commodities = sorted(filtered["commodity"].dropna().unique().tolist())
    return jsonify(commodities)

def fetch_mandi_prices_from_api(state=None, district=None, commodity=None, limit=50):
    """
    Fetches real-time APMC Mandi prices from the Government of India Open Data API (data.gov.in / Agmarknet).
    Resource ID: 9ef84268-d588-465a-a308-a864a43d0070
    """
    api_key = session.get("datagov_api_key") or os.environ.get("DATAGOV_API_KEY") or DATAGOV_API_KEY
    if not api_key or api_key == "YOUR_DATAGOV_API_KEY":
        api_key = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit
    }
    if state and state != "ALL":
        params["filters[state]"] = state
    if district and district != "ALL":
        params["filters[district]"] = district
    if commodity and commodity != "ALL":
        params["filters[commodity]"] = commodity

    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            records = data.get("records", [])
            if records:
                results = []
                for r in records:
                    results.append({
                        "market": r.get("market", "APMC Mandi"),
                        "commodity": r.get("commodity", commodity or "Crop"),
                        "variety": r.get("variety", "Standard"),
                        "min_price": float(r.get("min_price", 0) or 0),
                        "max_price": float(r.get("max_price", 0) or 0),
                        "modal_price": float(r.get("modal_price", 0) or 0),
                        "arrival_date": r.get("arrival_date", "Today"),
                        "source": "Live Agmarknet API"
                    })
                return results
    except Exception as e:
        print("Live Mandi API error:", e)

    return None

@app.route("/set-datagov-key", methods=["POST"])
def set_datagov_key():
    data = request.get_json() or {}
    key = data.get("api_key", "").strip()
    if key:
        session["datagov_api_key"] = key
        return jsonify({"status": "success", "message": "API Key saved successfully! Live Agmarknet API connected."})
    return jsonify({"status": "error", "message": "Please enter a valid API key"}), 400

@app.route("/get-prices-data")
def get_prices_data():
    state = request.args.get("state")
    district = request.args.get("district")
    commodity = request.args.get("commodity")

    if not state or not district:
        return jsonify({"error": "Missing parameters"})

    # 1. LIVE API FIRST: Try live Government Mandi API (Agmarknet via data.gov.in)
    api_records = fetch_mandi_prices_from_api(state=state, district=district, commodity=commodity)
    if api_records and len(api_records) > 0:
        return jsonify(api_records)

    # 2. Resilient Fallback to local APMC database if API is down, rate-limited, or slow
    query = (df_prices["state"] == state) & (df_prices["district"] == district)
    if commodity and commodity != "ALL":
        query = query & (df_prices["commodity"] == commodity)

    result = df_prices[query]

    if result.empty and commodity and commodity != "ALL":
        # Fallback search if commodity name differs slightly
        result = df_prices[(df_prices["state"] == state) & (df_prices["commodity"].str.contains(commodity, case=False, na=False))]

    if result.empty:
        return jsonify([])

    prices_list = []
    for _, row in result.iterrows():
        prices_list.append({
            "market": row['market'],
            "commodity": row['commodity'],
            "variety": row.get('variety', 'Standard'),
            "min_price": row['min_price'],
            "max_price": row['max_price'],
            "modal_price": row['modal_price'],
            "arrival_date": row.get('arrival_date', 'Daily Update'),
            "source": "APMC Mandi Feed"
        })

    return jsonify(prices_list)

COMMON_CROPS_MAP = {
    # Hindi
    "टमाटर": "Tomato", "आलू": "Potato", "प्याज": "Onion", "चावल": "Rice", "धान": "Paddy",
    "गेहूं": "Wheat", "मक्का": "Maize", "कपास": "Cotton", "सेब": "Apple", "मिर्च": "Chilli",
    "लहसुन": "Garlic", "अदरक": "Ginger", "हल्दी": "Turmeric", "सरसों": "Mustard", "चना": "Gram",
    # Telugu
    "టమోటా": "Tomato", "బంగాళాదుంప": "Potato", "ఉల్లిపాయ": "Onion", "వరి": "Paddy", "బియ్యం": "Rice",
    "గోధుమ": "Wheat", "మొక్కజొన్న": "Maize", "ప్రత్తి": "Cotton", "మిరపకాయ": "Chilli",
    # Tamil
    "தக்காளி": "Tomato", "உருளைக்கிழங்கு": "Potato", "வெங்காயம்": "Onion", "அரிசி": "Rice", "நெல்": "Paddy",
    "கோதுமை": "Wheat", "மக்காச்சோளம்": "Maize", "பருத்தி": "Cotton", "மிளகாய்": "Chilli",
    # Kannada
    "ಟೊಮೆಟೊ": "Tomato", "ಆಲೂಗಡ್ಡೆ": "Potato", "ಈರುಳ್ಳಿ": "Onion", "ಅಕ್ಕಿ": "Rice", "ಭತ್ತ": "Paddy",
    "ಗೋಧಿ": "Wheat", "ಮೆಕ್ಕೆಜೋಳ": "Maize", "ಹತ್ತಿ": "Cotton"
}

@app.route("/get-price")
def get_price():
    crop = request.args.get("crop", "").strip()
    lang = session.get("lang", "en")

    if not crop:
        return jsonify({"error": LANGUAGES.get(lang, LANGUAGES["en"])["crop_not_found"]})

    # If query is not in English, attempt translation to English for commodity lookup
    search_crop = COMMON_CROPS_MAP.get(crop, crop)
    if search_crop == crop and lang != "en":
        try:
            trans = translate_text(crop, "en")
            if trans:
                search_crop = trans
        except Exception:
            search_crop = crop

    # 1. LIVE API FIRST: Query live Agmarknet / data.gov.in API
    api_records = fetch_mandi_prices_from_api(commodity=search_crop, limit=1)
    if api_records and len(api_records) > 0:
        record = api_records[0]
        crop_name = record["commodity"]
        market_name = record["market"]
        price = record["modal_price"]
        district = "APMC Mandi"
        if lang != "en":
            try:
                crop_name = translate_text(crop_name, lang)
                market_name = translate_text(market_name, lang)
            except Exception:
                pass
        return jsonify({
            "crop": crop_name,
            "price": price,
            "market": market_name,
            "district": district,
            "source": "Live Agmarknet API"
        })

    # Filter matching crop from CSV fallback (matching user input or translated name)
    result = df_prices[
        df_prices["commodity"].str.contains(crop, case=False, na=False) |
        df_prices["commodity"].str.contains(search_crop, case=False, na=False)
    ]

    if result.empty:
        return jsonify({"error": LANGUAGES.get(lang, LANGUAGES["en"])["crop_not_found"]})

    # ✅ Get latest record safely
    latest = result.iloc[-1]

    crop_name = latest["commodity"]
    market_name = latest["market"]
    district = latest["district"]
    price = latest["modal_price"]

    # ✅ Translate if needed
    if lang != "en":
        try:
            crop_name = translate_text(crop_name, lang)
            market_name = translate_text(market_name, lang)
            district = translate_text(district, lang)
        except:
            pass

    return jsonify({
        "crop": crop_name,
        "price": price,
        "market": market_name,
        "district": district
    })

# ---------------- Schemes ----------------
@app.route("/schemes")
def schemes():
    lang = session.get("lang", "en")

    # State Schemes Dictionary with official portal links and clear descriptions
    schemes_data = {
        "Andhra Pradesh": [
            {
                "name": "YSR Rythu Bharosa",
                "desc": "Financial assistance of ₹13,500 per year for farmer families including tenant farmers.",
                "link": "https://ysrrythubharosa.ap.gov.in/"
            },
            {
                "name": "Dr. YSR Free Crop Insurance",
                "desc": "100% state-funded crop insurance scheme for notified crops against natural disasters.",
                "link": "https://karshak.ap.gov.in/"
            },
            {
                "name": "YSR Sunna Vaddi Panta Runalu",
                "desc": "Zero percent interest crop loans up to ₹1 Lakh for farmers who repay loans promptly.",
                "link": "https://apagrisnet.ap.gov.in/"
            },
            {
                "name": "YSR Yantra Seva Pathakam",
                "desc": "Subsidized farm machinery, tractors, and combine harvesters via Community Hiring Centers.",
                "link": "https://apagrisnet.ap.gov.in/"
            }
        ],
        "Telangana": [
            {
                "name": "Rythu Bandhu Scheme",
                "desc": "Agriculture investment support of ₹10,000 per acre per year directly into bank accounts.",
                "link": "https://rythubandhu.telangana.gov.in/"
            },
            {
                "name": "Rythu Bima Yojana",
                "desc": "Farmers group life insurance providing ₹5.00 Lakh financial relief to farmer families.",
                "link": "https://rythubima.telangana.gov.in/"
            },
            {
                "name": "Crop Loan Waiver (Runa Mafi)",
                "desc": "State government agricultural loan waiver scheme supporting distressed farming households.",
                "link": "https://clw.telangana.gov.in/"
            }
        ],
        "Tamil Nadu": [
            {
                "name": "Uzhavar Santhai Scheme",
                "desc": "Direct farmers markets to eliminate middlemen and ensure remunerative prices for farmers.",
                "link": "https://tnhorticulture.tn.gov.in/"
            },
            {
                "name": "Kalaignarin Integrated Agriculture Development",
                "desc": "Comprehensive village development transforming fallow land into productive cultivable land.",
                "link": "https://agritech.tnau.ac.in/"
            },
            {
                "name": "Free Power Supply Scheme",
                "desc": "Free continuous electricity for agricultural pump sets and irrigation borewells.",
                "link": "https://www.tnebnet.org/"
            }
        ],
        "Karnataka": [
            {
                "name": "Raitha Siri Yojane",
                "desc": "Direct incentive of ₹10,000 per hectare for farmers cultivating minor millets.",
                "link": "https://raitamitra.karnataka.gov.in/"
            },
            {
                "name": "Krishi Bhagya Scheme",
                "desc": "Support for farm ponds, polythene lining, diesel pumps, and micro-irrigation systems.",
                "link": "https://raitamitra.karnataka.gov.in/"
            },
            {
                "name": "Pashu Bhagya Scheme",
                "desc": "Subsidized loans and insurance for cattle, sheep, goat, and dairy farming.",
                "link": "https://ahvs.karnataka.gov.in/"
            }
        ],
        "Kerala": [
            {
                "name": "Karshaka Welfare Scheme",
                "desc": "Pensions, health assistance, and social security benefits for registered farmers.",
                "link": "https://keralaagriculture.gov.in/"
            },
            {
                "name": "Subhiksha Keralam Scheme",
                "desc": "Statewide agricultural food self-reliance mission and wasteland farming support.",
                "link": "https://keralaagriculture.gov.in/"
            },
            {
                "name": "State Crop Insurance Scheme",
                "desc": "Comprehensive crop loss protection covering 25 notified major crops.",
                "link": "https://aims.kerala.gov.in/"
            }
        ],
        "Maharashtra": [
            {
                "name": "Namo Shetkari Mahasanman Nidhi Yojana",
                "desc": "Annual financial grant of ₹6,000 to farmer families in addition to PM-KISAN.",
                "link": "https://mahadbt.maharashtra.gov.in/"
            },
            {
                "name": "Mahatma Jyotirao Phule Shetkari Karjmukti Yojana",
                "desc": "Comprehensive agricultural loan waiver providing debt relief up to ₹2 Lakh.",
                "link": "https://mjpsky.maharashtra.gov.in/"
            },
            {
                "name": "Magel Tyala Shet Tale (Farm Ponds on Demand)",
                "desc": "100% subsidy assistance for constructing individual on-farm water storage ponds.",
                "link": "https://krishi.maharashtra.gov.in/"
            }
        ],
        "Uttar Pradesh": [
            {
                "name": "UP Kisan Karj Mafi Yojana",
                "desc": "Farm loan waiver and credit relief for small and marginal cultivators across UP.",
                "link": "https://upkisankarjrahat.upsdc.gov.in/"
            },
            {
                "name": "Mukhyamantri Krishak Durghatna Kalyan Yojana",
                "desc": "Accidental life insurance coverage providing ₹5 Lakh financial aid to farmer families.",
                "link": "https://upagriculture.com/"
            },
            {
                "name": "Mukhyamantri Khet Suraksha Yojana",
                "desc": "Subsidized solar fencing to safeguard standing crops from stray and wild animals.",
                "link": "https://upagriculture.com/"
            }
        ],
        "Punjab": [
            {
                "name": "Pani Bachao Paisa Kamao",
                "desc": "Direct cash incentives for farmers saving ground water and electricity in paddy cultivation.",
                "link": "https://agripb.punjab.gov.in/"
            },
            {
                "name": "Crop Residue Management (CRM) Subsidy",
                "desc": "Up to 80% subsidy on Happy Seeders, Super Seeders, and Mulchers for stubble management.",
                "link": "https://agrimachinerypb.com/"
            },
            {
                "name": "Crop Diversification Mission",
                "desc": "Financial assistance to transition from water-intensive paddy to pulses, maize, and cotton.",
                "link": "https://agripb.punjab.gov.in/"
            }
        ],
        "Haryana": [
            {
                "name": "Bhavantar Bharpayee Yojana",
                "desc": "Direct price deficit compensation protecting horticulture farmers against market crashes.",
                "link": "https://fasal.haryana.gov.in/"
            },
            {
                "name": "Meri Fasal Mera Byora",
                "desc": "Single-window portal for crop procurement, MSP payments, subsidies, and compensation.",
                "link": "https://fasal.haryana.gov.in/"
            },
            {
                "name": "Mera Pani Meri Virasat",
                "desc": "Financial incentive of ₹7,000 per acre for replacing paddy with alternative crops.",
                "link": "https://fasal.haryana.gov.in/"
            }
        ],
        "Madhya Pradesh": [
            {
                "name": "Mukhyamantri Kisan Kalyan Yojana",
                "desc": "Direct income support of ₹6,000 per year per farmer family from the state government.",
                "link": "https://saara.mp.gov.in/"
            },
            {
                "name": "Bhavantar Bhugtan Yojana",
                "desc": "Price gap financing compensating farmers when market price falls below MSP.",
                "link": "https://mpkrishi.mp.gov.in/"
            },
            {
                "name": "Mukhyamantri Solar Pump Yojana",
                "desc": "Up to 90% subsidy on off-grid solar agricultural water pumping systems.",
                "link": "https://cmsolarpump.mp.gov.in/"
            }
        ],
        "Rajasthan": [
            {
                "name": "Rajasthan Krishi Rin Mafi Yojana",
                "desc": "Complete agricultural debt waiver for cooperative and commercial bank farm loans.",
                "link": "https://krishi.rajasthan.gov.in/"
            },
            {
                "name": "Tarbandi Yojana",
                "desc": "50% grant up to ₹40,000 for barbed-wire farm fencing against stray cattle.",
                "link": "https://rajkisan.rajasthan.gov.in/"
            },
            {
                "name": "PM KUSUM Solar Pump Scheme",
                "desc": "60% to 80% capital subsidy for installing standalone solar agricultural pumps.",
                "link": "https://energy.rajasthan.gov.in/"
            }
        ],
        "Gujarat": [
            {
                "name": "Saat Pagla Khedut Kalyan Na",
                "desc": "7-pillar agricultural welfare mission covering cow rearing, tools, fencing, and storage.",
                "link": "https://ikhedut.gujarat.gov.in/"
            },
            {
                "name": "Mukhyamantri Kisan Sahay Yojana",
                "desc": "Zero-premium state crop damage compensation for drought, excess rain, and unseasonal rainfall.",
                "link": "https://ikhedut.gujarat.gov.in/"
            },
            {
                "name": "Suryashakti Kisan Yojana (SKY)",
                "desc": "Grid-connected solar farm power allowing farmers to generate irrigation power and sell surplus.",
                "link": "https://ugvcl.com/"
            }
        ],
        "Bihar": [
            {
                "name": "Bihar Rajya Fasal Sahayata Yojana",
                "desc": "Free state crop assistance providing up to ₹10,000/ha for flood, drought, and crop loss.",
                "link": "https://pacsonline.bih.nic.in/"
            },
            {
                "name": "Diesel Anudan (Fuel Subsidy) Scheme",
                "desc": "Direct diesel fuel subsidy of ₹750/acre for irrigation during dry spells.",
                "link": "https://dbtagriculture.bihar.gov.in/"
            },
            {
                "name": "Krishi Yantrikaran Yojana",
                "desc": "Heavy subsidies up to 80% on 75+ agricultural implements, rotavators, and tractors.",
                "link": "https://farmech.bih.nic.in/"
            }
        ],
        "West Bengal": [
            {
                "name": "Krishak Bandhu (Natun)",
                "desc": "Annual assured direct grant of ₹10,000/acre plus ₹2 Lakh family life insurance.",
                "link": "https://krishakbandhu.wb.gov.in/"
            },
            {
                "name": "Bangla Shasya Bima (BSB)",
                "desc": "Fully state-funded crop insurance covering 100% of farmer premium for major crops.",
                "link": "https://banglashasyabima.net/"
            },
            {
                "name": "Farm Mechanization & Chashi Kalyan",
                "desc": "Subsidies for Custom Hiring Centers (CHCs) and small farm mechanization equipment.",
                "link": "https://matirkatha.gov.in/"
            }
        ],
        "Odisha": [
            {
                "name": "KALIA Scheme (Krushak Assistance)",
                "desc": "Financial assistance of ₹10,000/year for small/marginal farmers and landless farm laborers.",
                "link": "https://kalia.odisha.gov.in/"
            },
            {
                "name": "Balaram Yojana",
                "desc": "Agricultural credit and collateral-free loan support for landless tenant farmers via JLGs.",
                "link": "https://agri.odisha.gov.in/"
            },
            {
                "name": "Jalanidhi Micro-Irrigation Scheme",
                "desc": "Heavy capital subsidy up to 70% for shallow tube wells, bore wells, and river lifts.",
                "link": "https://agrisnetodisha.ori.nic.in/"
            }
        ]
    }

    news_list = []
    try:
        feed = feedparser.parse("https://krishijagran.com/feeds/?cat=17")
        if feed.entries:
            news_list = [entry.title for entry in feed.entries[:5]]
    except Exception as e:
        print("RSS Feed error:", e)

    # ✅ Translate dynamically
    if lang != "en":
        translated = {}
        for state, scheme_list in schemes_data.items():
            translated_state = translate_text(state, lang)
            translated[translated_state] = []
            for s in scheme_list:
                translated[translated_state].append({
                    "name": translate_text(s["name"], lang),
                    "desc": translate_text(s["desc"], lang),
                    "link": s["link"]
                })
        schemes_data = translated
        if news_list:
            news_list = [translate_text(n, lang) for n in news_list]

    return render_template("schemes.html", schemes_json=schemes_data, news_list=news_list)

# ---------------- Weather ----------------

@app.route("/weather")
def weather():
    return render_template("weather.html")

@app.route("/get-weather")
def get_weather_api():
    city = request.args.get("city")

    if not city:
        return jsonify({"error": "City not provided"})

    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(current_url, timeout=6)
        current = resp.json()

        # Fallback to geocoding if direct city lookup fails
        if str(current.get("cod")) != "200":
            try:
                geo_url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(city)}&countrycodes=in&limit=1"
                geo_res = requests.get(geo_url, headers={"User-Agent": "FarmerAssistant/1.0"}, timeout=4)
                if geo_res.status_code == 200 and geo_res.json():
                    lat = geo_res.json()[0]['lat']
                    lon = geo_res.json()[0]['lon']
                    coord_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
                    coord_resp = requests.get(coord_url, timeout=6)
                    current = coord_resp.json()
            except Exception:
                pass

        if str(current.get("cod")) != "200":
            return jsonify({"error": current.get("message", "City not found")}), 404

        return jsonify({
            "city": current.get("name", city),
            "temp": current["main"]["temp"],
            "humidity": current["main"]["humidity"],
            "condition": current["weather"][0]["description"]
        })

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve weather: {str(e)}"}), 500

# ---------------- Farmer AI ----------------

@app.route("/farmer-ai")
def farmer_ai():
    return render_template("farmer_ai.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_msg = data.get("message", "")
    lang = data.get("lang") or session.get("lang", "en")

    reply = chatbot_reply(user_msg, lang)

    return jsonify({
        "status": "success",
        "reply": reply
    })

# ---------------- Soil ----------------

@app.route("/soil")
def soil():
    return render_template("soil.html")

@app.route("/get-soil")
def get_soil():
    soil_type = request.args.get("soil", "").strip()

    # Normalization map for soil types to handle all variations
    soil_map = {
        "black (regur)": "Black",
        "desert / arid": "Desert",
        "mountain / forest": "Mountain",
        "peaty / marshy": "Peaty",
        "saline / alkaline": "Saline",
        "regur": "Black",
        "arid": "Desert",
        "forest": "Mountain",
        "marshy": "Peaty",
        "alkaline": "Saline"
    }
    lookup_soil = soil_map.get(soil_type.lower(), soil_type)

    df_soil = pd.read_csv("data/soil_data.csv")
    result = df_soil[df_soil["soil_type"].str.lower() == lookup_soil.lower()]

    if result.empty:
        result = df_soil[df_soil["soil_type"].str.lower().str.contains(lookup_soil.lower(), na=False)]

    if result.empty:
        return jsonify({"error": "Soil data not found"})

    row = result.iloc[0]

    return jsonify({
        "advantages": row["advantages"],
        "limitations": row["limitations"],
        "best_crops": row["best_crops"],
        "fertilizer_tips": row["fertilizer_tips"],
        "pest_management": row["pest_management"]
    })

# ---------------- Calendar ----------------

@app.route("/calendar")
def calendar():
    return render_template("calendar.html")

@app.route("/get-calendar")
def get_calendar():
    state = request.args.get("state")

    df_cal = pd.read_csv("data/farming_calendar.csv")
    result = df_cal[df_cal["state"] == state]

    if result.empty:
        return jsonify({"error": "No calendar data found"})

    return jsonify(result.to_dict(orient="records"))

# ---------------- Run ----------------

if __name__ == "__main__":
    app.run(debug=True)
