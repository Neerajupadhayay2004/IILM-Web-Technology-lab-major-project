"""
Phishing Website Detection - Flask API
Full-stack ML-powered web application
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import time
from datetime import datetime
from feature_extractor import extract_features, features_to_vector

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join('model', 'phishing_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model not found. Run train_model.py first!")
    model = None
    scaler = None

# ── Feature names for display ─────────────────────────────────────────────────
FEATURE_NAMES = [
    'url_length', 'hostname_length', 'dots_in_hostname', 'hyphens_in_hostname',
    'digits_in_hostname', 'path_length', 'slashes_in_path', 'query_length',
    'query_params', 'special_chars', 'uses_https', 'has_at_symbol',
    'has_double_slash', 'has_ip_address', 'has_port', 'subdomain_count',
    'registered_domain_length', 'suspicious_tld', 'whitelisted_domain',
    'domain_entropy', 'brand_in_subdomain', 'brand_in_path',
    'phishing_keywords', 'keyword_in_domain', 'hex_encoding',
    'excessive_subdomains', 'is_shortened', 'digit_ratio',
    'hyphen_in_registered_domain', 'repeated_chars'
]

FEATURE_LABELS = {
    'url_length': 'URL Length',
    'hostname_length': 'Hostname Length',
    'dots_in_hostname': 'Dots in Hostname',
    'hyphens_in_hostname': 'Hyphens in Hostname',
    'digits_in_hostname': 'Digits in Hostname',
    'path_length': 'Path Length',
    'slashes_in_path': 'Slashes in Path',
    'query_length': 'Query String Length',
    'query_params': 'Query Parameters',
    'special_chars': 'Special Characters',
    'uses_https': 'Uses HTTPS',
    'has_at_symbol': 'Has @ Symbol',
    'has_double_slash': 'Has Double Slash',
    'has_ip_address': 'Has IP Address',
    'has_port': 'Non-standard Port',
    'subdomain_count': 'Subdomain Count',
    'registered_domain_length': 'Domain Name Length',
    'suspicious_tld': 'Suspicious TLD',
    'whitelisted_domain': 'Whitelisted Domain',
    'domain_entropy': 'Domain Entropy',
    'brand_in_subdomain': 'Brand in Subdomain',
    'brand_in_path': 'Brand in Path',
    'phishing_keywords': 'Phishing Keywords',
    'keyword_in_domain': 'Keywords in Domain',
    'hex_encoding': 'Hex Encoding Count',
    'excessive_subdomains': 'Excessive Subdomains',
    'is_shortened': 'URL Shortener',
    'digit_ratio': 'Digit Ratio',
    'hyphen_in_registered_domain': 'Hyphens in Domain',
    'repeated_chars': 'Repeated Characters',
}

# ── In-memory scan history ────────────────────────────────────────────────────
scan_history = []
MAX_HISTORY = 50


def analyze_risk_factors(features_dict, prediction, confidence):
    """Generate human-readable risk analysis"""
    risks = []
    safe = []

    if features_dict.get('uses_https') == 0:
        risks.append({'icon': '🔓', 'text': 'No HTTPS encryption – data sent in plaintext'})
    else:
        safe.append({'icon': '🔒', 'text': 'HTTPS encryption enabled'})

    if features_dict.get('has_ip_address') == 1:
        risks.append({'icon': '🔢', 'text': 'IP address used instead of domain name'})

    if features_dict.get('suspicious_tld') == 1:
        risks.append({'icon': '⚠️', 'text': 'Suspicious free TLD (.tk, .ml, .xyz, etc.)'})

    if features_dict.get('has_at_symbol') == 1:
        risks.append({'icon': '📧', 'text': '@ symbol in URL – browser ignores everything before it'})

    if features_dict.get('brand_in_subdomain') == 1:
        risks.append({'icon': '🏷️', 'text': 'Trusted brand name used in subdomain to deceive'})

    if features_dict.get('brand_in_path') == 1:
        risks.append({'icon': '📂', 'text': 'Brand name appears in URL path (not domain)'})

    if features_dict.get('phishing_keywords', 0) > 2:
        risks.append({'icon': '🎣', 'text': f"Multiple phishing keywords detected ({features_dict['phishing_keywords']})"})
    elif features_dict.get('phishing_keywords', 0) > 0:
        risks.append({'icon': '🔑', 'text': 'Phishing-related keywords found in URL'})

    if features_dict.get('excessive_subdomains') == 1:
        risks.append({'icon': '🌐', 'text': 'Excessive subdomains – common obfuscation tactic'})

    if features_dict.get('is_shortened') == 1:
        risks.append({'icon': '✂️', 'text': 'URL shortener detected – hides real destination'})

    if features_dict.get('has_port') == 1:
        risks.append({'icon': '🔌', 'text': 'Non-standard port number in URL'})

    if features_dict.get('hex_encoding', 0) > 3:
        risks.append({'icon': '💻', 'text': f"Heavy URL encoding ({features_dict['hex_encoding']} encoded chars)"})

    if features_dict.get('domain_entropy', 0) > 3.5:
        risks.append({'icon': '🎲', 'text': 'Domain name appears random/generated'})

    if features_dict.get('whitelisted_domain') == 1:
        safe.append({'icon': '✅', 'text': 'Domain is in trusted whitelist'})

    if features_dict.get('url_length', 0) > 75:
        risks.append({'icon': '📏', 'text': f"Unusually long URL ({features_dict['url_length']} chars)"})
    else:
        safe.append({'icon': '📏', 'text': 'URL length is normal'})

    return risks, safe


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main URL analysis endpoint"""
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url'].strip()
    if not url:
        return jsonify({'error': 'Empty URL'}), 400

    start_time = time.time()

    try:
        # Extract features
        features_dict = extract_features(url)
        feature_vector = features_to_vector(features_dict)
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        phishing_prob = float(probabilities[1])
        legitimate_prob = float(probabilities[0])

        # Determine risk level
        if phishing_prob >= 0.85:
            risk_level = 'CRITICAL'
            risk_color = '#ff2d55'
        elif phishing_prob >= 0.65:
            risk_level = 'HIGH'
            risk_color = '#ff6b35'
        elif phishing_prob >= 0.45:
            risk_level = 'MEDIUM'
            risk_color = '#f7c59f'
        elif phishing_prob >= 0.25:
            risk_level = 'LOW'
            risk_color = '#a8e6cf'
        else:
            risk_level = 'SAFE'
            risk_color = '#3dd68c'

        # Analyze risk factors
        risks, safe_factors = analyze_risk_factors(features_dict, prediction, phishing_prob)

        # Build feature detail list for display
        feature_details = []
        for name in FEATURE_NAMES:
            val = features_dict.get(name, 0)
            feature_details.append({
                'name': name,
                'label': FEATURE_LABELS.get(name, name),
                'value': val,
                'type': 'bool' if name in [
                    'uses_https', 'has_at_symbol', 'has_double_slash',
                    'has_ip_address', 'has_port', 'suspicious_tld',
                    'whitelisted_domain', 'brand_in_subdomain', 'brand_in_path',
                    'excessive_subdomains', 'is_shortened', 'repeated_chars'
                ] else 'number'
            })

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        result = {
            'url': url,
            'prediction': int(prediction),
            'label': 'Phishing' if prediction == 1 else 'Legitimate',
            'phishing_probability': round(phishing_prob * 100, 2),
            'legitimate_probability': round(legitimate_prob * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risks,
            'safe_factors': safe_factors,
            'features': feature_details,
            'analysis_time_ms': elapsed_ms,
            'timestamp': datetime.now().isoformat(),
        }

        # Add to history
        history_entry = {
            'url': url[:80] + '...' if len(url) > 80 else url,
            'label': result['label'],
            'risk_level': risk_level,
            'probability': round(phishing_prob * 100, 1),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
        }
        scan_history.insert(0, history_entry)
        if len(scan_history) > MAX_HISTORY:
            scan_history.pop()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    """Analyze multiple URLs at once"""
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({'error': 'No URLs provided'}), 400

    urls = data['urls']
    if not isinstance(urls, list) or len(urls) == 0:
        return jsonify({'error': 'Invalid URL list'}), 400

    if len(urls) > 50:
        return jsonify({'error': 'Maximum 50 URLs per batch'}), 400

    results = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            features_dict = extract_features(url)
            feature_vector = features_to_vector(features_dict)
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            prob = float(model.predict_proba(X_scaled)[0][1])

            if prob >= 0.85:
                risk = 'CRITICAL'
            elif prob >= 0.65:
                risk = 'HIGH'
            elif prob >= 0.45:
                risk = 'MEDIUM'
            elif prob >= 0.25:
                risk = 'LOW'
            else:
                risk = 'SAFE'

            results.append({
                'url': url,
                'label': 'Phishing' if prediction == 1 else 'Legitimate',
                'phishing_probability': round(prob * 100, 2),
                'risk_level': risk,
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results, 'total': len(results)})


@app.route('/api/history', methods=['GET'])
def get_history():
    """Return scan history"""
    return jsonify({'history': scan_history, 'total': len(scan_history)})


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear scan history"""
    scan_history.clear()
    return jsonify({'message': 'History cleared'})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return statistics about scans performed"""
    if not scan_history:
        return jsonify({'total': 0, 'phishing': 0, 'legitimate': 0, 'phishing_rate': 0})

    total = len(scan_history)
    phishing = sum(1 for s in scan_history if s['label'] == 'Phishing')
    legitimate = total - phishing

    return jsonify({
        'total': total,
        'phishing': phishing,
        'legitimate': legitimate,
        'phishing_rate': round(phishing / total * 100, 1) if total > 0 else 0,
        'critical': sum(1 for s in scan_history if s['risk_level'] == 'CRITICAL'),
        'high': sum(1 for s in scan_history if s['risk_level'] == 'HIGH'),
        'medium': sum(1 for s in scan_history if s['risk_level'] == 'MEDIUM'),
        'low': sum(1 for s in scan_history if s['risk_level'] == 'LOW'),
        'safe': sum(1 for s in scan_history if s['risk_level'] == 'SAFE'),
    })


@app.route('/api/features', methods=['POST'])
def get_features():
    """Return raw features for a URL (debugging/analysis)"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url'].strip()
    try:
        features = extract_features(url)
        return jsonify({'url': url, 'features': features})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scans_in_session': len(scan_history),
        'version': '1.0.0'
    })


if __name__ == '__main__':
    print("\n🛡️  Phishing Website Detection System")
    print("=" * 45)
    print("📡  Starting Flask server...")
    print("🌐  Visit: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
