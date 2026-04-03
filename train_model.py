"""
Train and save the phishing detection ML model.
Uses a Random Forest classifier with synthetic + rule-based training data.
"""

import numpy as np
import joblib
import os
from feature_extractor import extract_features, features_to_vector

# Suppress sklearn warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ── Synthetic Dataset ─────────────────────────────────────────────────────────

LEGITIMATE_URLS = [
    "https://www.google.com/search?q=python",
    "https://github.com/openai/gpt-4",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://stackoverflow.com/questions/1234567",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://twitter.com/elonmusk/status/123",
    "https://www.reddit.com/r/programming/comments/abc",
    "https://www.linkedin.com/in/username",
    "https://www.facebook.com/zuck",
    "https://www.microsoft.com/en-us/windows",
    "https://www.apple.com/iphone",
    "https://www.netflix.com/browse",
    "https://www.paypal.com/us/home",
    "https://www.ebay.com/itm/1234567",
    "https://www.dropbox.com/home",
    "https://slack.com/intl/en-us",
    "https://zoom.us/meeting",
    "https://notion.so/workspace",
    "https://figma.com/file/abc123",
    "https://stripe.com/docs",
    "https://shopify.com/login",
    "https://wordpress.com/create",
    "https://adobe.com/products/photoshop",
    "https://salesforce.com/crm",
    "https://www.yahoo.com/finance",
    "https://outlook.live.com/mail",
    "https://office.com/launch/word",
    "https://bing.com/search?q=ai",
    "https://www.instagram.com/explore",
    "https://docs.python.org/3/library/os.html",
    "https://developer.mozilla.org/en-US/docs/Web",
    "https://www.cloudflare.com/learning/",
    "https://aws.amazon.com/s3",
    "https://cloud.google.com/storage",
    "https://azure.microsoft.com/en-us",
    "https://www.twitch.tv/directory",
    "https://discord.com/channels/@me",
    "https://trello.com/b/abc123",
    "https://asana.com/projects",
    "https://hubspot.com/products/crm",
    "https://mailchimp.com/campaigns",
    "https://www.coursera.org/learn/python",
    "https://www.udemy.com/course/python",
    "https://medium.com/@author/article",
    "https://dev.to/post/article",
    "https://www.w3schools.com/html",
    "https://css-tricks.com/flexbox",
    "https://reactjs.org/docs/getting-started",
    "https://vuejs.org/guide/introduction",
]

PHISHING_URLS = [
    "http://paypal-secure-login.tk/verify/account",
    "http://192.168.1.1/login/paypal/verify",
    "https://secure-paypal-login.xyz/update/now",
    "http://apple-id-verify.ml/signin/confirm",
    "http://microsoft-account-alert.ga/login",
    "http://amazon-order-update.cf/signin",
    "http://www.paypal.com.secure-login.tk/",
    "http://login-facebook.com.phishing.xyz/",
    "https://google-account-verify.click/secure",
    "http://netfl1x-billing.ml/update/payment",
    "http://bankofamerica-secure.xyz/login",
    "http://chase-bank-verify.tk/account/confirm",
    "http://hsbc.com.verify-account.ml/login",
    "http://instagram-verify.ga/checkpoint/login",
    "http://twitter-suspend-appeal.cf/unlock",
    "http://dropbox-storage-full.xyz/upgrade",
    "http://steam-free-games.tk/claim/gift",
    "http://coinbase-wallet-verify.ml/secure",
    "http://binance-bonus.ga/claim/1000usdt",
    "http://verify-login-secure.xyz/paypal/account",
    "http://free-iphone-winner.tk/claim/now",
    "http://your-package-update.ml/track/123456789",
    "http://irs-refund-pending.cf/claim/refund",
    "http://linkedin-premium-free.xyz/activate",
    "http://zoom-meeting-invite.tk/join/urgent",
    "https://secure.paypa1.com/login",
    "http://paypal.com-secure-login.net/verify",
    "http://signin.amazon.com.update-billing.tk/",
    "http://apple.com-id-locked.ml/unlock/account",
    "http://microsofft-support.xyz/remote-access",
    "http://googgle-docs-share.tk/view/document",
    "http://netflixx-account.ga/suspend/recover",
    "http://facebok-login.cf/recover/account",
    "http://twittter-locked.xyz/appeal/unlock",
    "http://instagramm-verify.ml/checkpoint",
    "http://verify-your-account-now.tk/login",
    "http://account-suspended-urgent.xyz/restore",
    "http://click-here-to-win-prize.tk/claim",
    "http://limited-offer-expires-soon.ml/buy",
    "http://your-account-has-been-compromised.ga/",
    "http://secure-update-required.xyz/banking",
    "http://unusual-signin-detected.tk/verify",
    "http://action-required-immediately.ml/login",
    "http://confirm-your-details-now.ga/secure",
    "http://suspicious-activity-alert.cf/review",
    "https://paypal-limited-access.click/restore",
    "http://update-your-payment-method.xyz/billing",
    "http://account-verification-needed.tk/verify",
    "http://security-check-required.ml/confirm",
    "http://log-in-to-verify.ga/account/secure",
]


def generate_noisy_features(url, label, n_variants=5):
    """Generate slightly varied feature vectors for data augmentation"""
    base_features = features_to_vector(extract_features(url))
    variants = [base_features]
    
    for _ in range(n_variants):
        noisy = base_features.copy()
        # Add small random noise to continuous features
        for i in range(len(noisy)):
            if isinstance(noisy[i], float):
                noisy[i] = max(0, noisy[i] + np.random.normal(0, 0.01))
        variants.append(noisy)
    
    return variants


def build_dataset():
    """Build training dataset from URL lists"""
    X, y = [], []
    
    print("Extracting features from legitimate URLs...")
    for url in LEGITIMATE_URLS:
        try:
            variants = generate_noisy_features(url, 0, n_variants=8)
            X.extend(variants)
            y.extend([0] * len(variants))
        except Exception as e:
            print(f"  Error on {url}: {e}")
    
    print("Extracting features from phishing URLs...")
    for url in PHISHING_URLS:
        try:
            variants = generate_noisy_features(url, 1, n_variants=8)
            X.extend(variants)
            y.extend([1] * len(variants))
        except Exception as e:
            print(f"  Error on {url}: {e}")
    
    return np.array(X), np.array(y)


def train_and_save_model(output_dir='model'):
    """Train ensemble model and save to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🔍 Building dataset...")
    X, y = build_dataset()
    print(f"   Total samples: {len(X)} ({sum(y==0)} legitimate, {sum(y==1)} phishing)")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n🤖 Training ensemble model...")
    
    # Build Voting Ensemble
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[2, 1]
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test_scaled)
    y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"   ROC-AUC Score: {auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='f1')
    print(f"   Cross-val F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save model and scaler
    model_path = os.path.join(output_dir, 'phishing_model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    joblib.dump(ensemble, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Feature importance from RF component
    rf_model = ensemble.named_estimators_['rf']
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
    
    importances = rf_model.feature_importances_
    top_features = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n🏆 Top 10 Feature Importances:")
    for name, imp in top_features:
        bar = '█' * int(imp * 100)
        print(f"   {name:<30} {bar} {imp:.4f}")
    
    return ensemble, scaler


if __name__ == '__main__':
    train_and_save_model()
