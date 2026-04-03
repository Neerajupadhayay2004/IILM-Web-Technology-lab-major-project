"""
Phishing Website Feature Extractor
Extracts 30 features from URLs for ML classification
"""

import re
import math
from urllib.parse import urlparse
from collections import Counter


# Known legitimate domains (whitelist)
WHITELIST = {
    'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com',
    'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com', 'github.com',
    'wikipedia.org', 'reddit.com', 'netflix.com', 'paypal.com', 'ebay.com',
    'yahoo.com', 'bing.com', 'outlook.com', 'office.com', 'dropbox.com',
    'stripe.com', 'shopify.com', 'wordpress.com', 'adobe.com', 'salesforce.com',
    'zoom.us', 'slack.com', 'notion.so', 'figma.com', 'vercel.app'
}

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click', '.link',
    '.work', '.live', '.site', '.online', '.website', '.store', '.tech',
    '.pw', '.cc', '.info', '.biz', '.name', '.mobi', '.tel'
}

# Brand names commonly targeted in phishing
TARGETED_BRANDS = [
    'paypal', 'apple', 'google', 'microsoft', 'amazon', 'facebook',
    'instagram', 'twitter', 'netflix', 'bank', 'chase', 'wellsfargo',
    'citibank', 'hsbc', 'ebay', 'dropbox', 'linkedin', 'yahoo',
    'outlook', 'office365', 'steam', 'coinbase', 'binance', 'metamask'
]

# Phishing keywords
PHISHING_KEYWORDS = [
    'login', 'signin', 'sign-in', 'logon', 'verify', 'verification',
    'secure', 'security', 'update', 'confirm', 'account', 'password',
    'passwd', 'credential', 'banking', 'wallet', 'support', 'helpdesk',
    'service', 'alert', 'urgent', 'suspend', 'blocked', 'limited',
    'validate', 'authenticate', 'authorize', 'recover', 'restore',
    'click', 'free', 'winner', 'prize', 'gift', 'bonus', 'reward'
]


def shannon_entropy(s):
    """Calculate Shannon entropy of a string"""
    if not s:
        return 0
    freq = Counter(s)
    length = len(s)
    return -sum((count/length) * math.log2(count/length) for count in freq.values())


def extract_features(url):
    """
    Extract 30 features from a URL for phishing detection.
    Returns a dict of feature_name -> value
    """
    features = {}
    
    # Normalize URL
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
        query = parsed.query or ''
        full_url = url.lower()
        domain_lower = hostname.lower()
    except Exception:
        # Return default high-risk features if parsing fails
        return {f'f{i}': 1 for i in range(30)}

    # ─── URL Structure Features ───────────────────────────────────────
    # F1: URL length (longer URLs are more suspicious)
    features['url_length'] = len(url)

    # F2: Hostname length
    features['hostname_length'] = len(hostname)

    # F3: Number of dots in hostname
    features['dots_in_hostname'] = hostname.count('.')

    # F4: Number of hyphens in hostname
    features['hyphens_in_hostname'] = hostname.count('-')

    # F5: Number of digits in hostname
    features['digits_in_hostname'] = sum(c.isdigit() for c in hostname)

    # F6: Path length
    features['path_length'] = len(path)

    # F7: Number of slashes in path
    features['slashes_in_path'] = path.count('/')

    # F8: Query string length
    features['query_length'] = len(query)

    # F9: Number of query parameters
    features['query_params'] = len(query.split('&')) if query else 0

    # F10: Count of special characters (@, !, ?, =, &, #)
    special_chars = sum(url.count(c) for c in ['@', '!', '?', '=', '&', '#', '%'])
    features['special_chars'] = special_chars

    # ─── Protocol & Security Features ────────────────────────────────
    # F11: Uses HTTPS (1=yes, 0=no)
    features['uses_https'] = 1 if url.startswith('https://') else 0

    # F12: Has @ symbol (common phishing trick)
    features['has_at_symbol'] = 1 if '@' in url else 0

    # F13: Has double slash after protocol
    features['has_double_slash'] = 1 if '//' in url[8:] else 0

    # F14: Has IP address instead of domain
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features['has_ip_address'] = 1 if re.search(ip_pattern, hostname) else 0

    # F15: Port number in URL
    features['has_port'] = 1 if parsed.port and parsed.port not in (80, 443) else 0

    # ─── Domain-Based Features ───────────────────────────────────────
    # F16: Domain token count (subdomains)
    domain_parts = hostname.split('.')
    features['subdomain_count'] = max(0, len(domain_parts) - 2)

    # F17: Registered domain length (second-level domain)
    registered_domain = domain_parts[-2] if len(domain_parts) >= 2 else hostname
    features['registered_domain_length'] = len(registered_domain)

    # F18: TLD is suspicious
    tld = '.' + domain_parts[-1] if domain_parts else ''
    features['suspicious_tld'] = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

    # F19: Domain is in whitelist
    # Check last two parts of domain
    base_domain = '.'.join(domain_parts[-2:]) if len(domain_parts) >= 2 else hostname
    features['whitelisted_domain'] = 1 if base_domain.lower() in WHITELIST else 0

    # F20: Entropy of domain name (high entropy = random-looking = suspicious)
    features['domain_entropy'] = round(shannon_entropy(registered_domain), 4)

    # ─── Brand & Keyword Features ────────────────────────────────────
    # F21: Brand name in subdomain (e.g., paypal.attacker.com)
    subdomain = '.'.join(domain_parts[:-2]) if len(domain_parts) > 2 else ''
    features['brand_in_subdomain'] = 1 if any(b in subdomain.lower() for b in TARGETED_BRANDS) else 0

    # F22: Brand name in path (not in domain)
    features['brand_in_path'] = 1 if any(b in path.lower() for b in TARGETED_BRANDS) else 0

    # F23: Phishing keyword count in URL
    features['phishing_keywords'] = sum(1 for kw in PHISHING_KEYWORDS if kw in full_url)

    # F24: Phishing keyword in domain
    features['keyword_in_domain'] = sum(1 for kw in PHISHING_KEYWORDS if kw in domain_lower)

    # ─── Obfuscation Features ────────────────────────────────────────
    # F25: Hex encoding in URL (%XX patterns)
    features['hex_encoding'] = len(re.findall(r'%[0-9a-fA-F]{2}', url))

    # F26: Multiple subdomains (more than 3 = suspicious)
    features['excessive_subdomains'] = 1 if features['subdomain_count'] > 2 else 0

    # F27: URL shortener patterns
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co', 'rebrand.ly', 'cutt.ly']
    features['is_shortened'] = 1 if any(s in full_url for s in shorteners) else 0

    # F28: Digit ratio in hostname (e.g., 123secure-paypal456.com)
    digit_ratio = features['digits_in_hostname'] / max(len(hostname), 1)
    features['digit_ratio'] = round(digit_ratio, 4)

    # F29: Hyphen ratio in registered domain
    hyphen_count = registered_domain.count('-')
    features['hyphen_in_registered_domain'] = hyphen_count

    # F30: Consecutive repeated characters (suspicious pattern)
    features['repeated_chars'] = 1 if re.search(r'(.)\1{3,}', hostname) else 0

    return features


def features_to_vector(features_dict):
    """Convert features dict to ordered numpy array"""
    FEATURE_ORDER = [
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
    return [features_dict.get(f, 0) for f in FEATURE_ORDER]
