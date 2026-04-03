# 🛡️ PhishGuard — Phishing Website Detection System

Live Demo: https://iilm-web-technology-lab-major-proje.vercel.app/


backend: https://iilm-web-technology-lab-major-project.onrender.com/

ML-powered phishing URL detection using Flask + scikit-learn ensemble classifier.

## Features
- **30-feature URL analysis** (structure, entropy, keywords, TLD, brand detection, etc.)
- **Ensemble ML model** (Random Forest + Gradient Boosting Voting Classifier)
- **Real-time single URL scan** with risk breakdown
- **Batch scanning** (up to 50 URLs at once)
- **Scan history** with session statistics
- **REST API** for integration
- **Dark-themed cybersecurity UI**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (one-time)
python train_model.py

# 3. Start the Flask server
python app.py

# 4. Open browser at http://127.0.0.1:5000
```

## API Endpoints

| Method | Endpoint       | Description                     |
|--------|----------------|---------------------------------|
| POST   | /api/analyze   | Analyze a single URL            |
| POST   | /api/batch     | Analyze up to 50 URLs at once   |
| POST   | /api/features  | Get raw features for a URL      |
| GET    | /api/history   | Get scan history                |
| DELETE | /api/history   | Clear scan history              |
| GET    | /api/stats     | Get session statistics          |
| GET    | /api/health    | Health check                    |

### Example: Single URL Analysis
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-secure-login.tk/verify"}'
```

### Example Response
```json
{
  "url": "http://paypal-secure-login.tk/verify",
  "prediction": 1,
  "label": "Phishing",
  "phishing_probability": 99.7,
  "legitimate_probability": 0.3,
  "risk_level": "CRITICAL",
  "risk_factors": [
    {"icon": "⚠️", "text": "Suspicious free TLD (.tk, .ml, .xyz, etc.)"},
    {"icon": "🏷️", "text": "Trusted brand name used in subdomain to deceive"},
    {"icon": "🔑", "text": "Phishing-related keywords found in URL"}
  ],
  "safe_factors": [],
  "features": [...],
  "analysis_time_ms": 12.4
}
```

## ML Features Extracted (30 total)

| Category | Features |
|----------|----------|
| URL Structure | Length, hostname length, dots, hyphens, digits, slashes, query params |
| Protocol | HTTPS usage, @ symbol, double slash, IP address, port number |
| Domain | Subdomains, TLD suspiciousness, whitelist check, entropy |
| Brand/Keywords | Brand in subdomain/path, phishing keyword count |
| Obfuscation | Hex encoding, URL shorteners, digit ratio, repeated chars |

## Project Structure
```
phishing-detector/
├── app.py                  # Flask application
├── feature_extractor.py    # URL feature extraction (30 features)
├── train_model.py          # ML model training
├── requirements.txt
├── model/
│   ├── phishing_model.pkl  # Trained ensemble model
│   └── scaler.pkl          # Feature scaler
└── templates/
    └── index.html          # Frontend UI
```
