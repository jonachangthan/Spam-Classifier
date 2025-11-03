"""Text preprocessing module for spam classifier."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def lowercase_normalize(text):
    """Convert text to lowercase.
    Example:
        "Ok lar... Joking wif u oni..." -> "ok lar... joking wif u oni..."
    """
    return str(text).lower()

def mask_contacts(text):
    """Replace URLs, emails, and phone numbers with tokens.
    Example:
        "Contact me at test@example.com" -> "contact me at <EMAIL>"
        "Visit https://example.com" -> "visit <URL>"
        "Call +1 415-555-1212" -> "call <PHONE>"
    """
    text = str(text).lower()
    
    # URL pattern
    url_pattern = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9]+(?:\.[a-zA-Z]{2,})+)'
    text = re.sub(url_pattern, '<URL>', text)
    
    # Email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    text = re.sub(email_pattern, '<EMAIL>', text)
    
    # Phone pattern - various formats
    phone_patterns = [
        r'\+\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}',  # International
        r'\d{3}[-\s]?\d{3}[-\s]?\d{4}',                  # US/Canada
        r'\d{4}[-\s]?\d{3}[-\s]?\d{3}',                  # Other formats
        r'\d{5}[-\s]?\d{6}',                             # Other formats
        r'\(\d{3}\)[-\s]?\d{3}[-\s]?\d{4}'              # (123) 456-7890
    ]
    for pattern in phone_patterns:
        text = re.sub(pattern, '<PHONE>', text)
    
    return text

def handle_numbers(text, keep_numbers=False):
    """Replace digit sequences with <NUM> unless keep_numbers is True.
    Example:
        "Text FA to 87121 to receive entry" -> "text fa to <NUM> to receive entry"
    """
    text = str(text).lower()
    if not keep_numbers:
        text = re.sub(r'\b\d+\b', '<NUM>', text)
    return text

def strip_punctuation(text):
    """Remove punctuation while keeping word chars, spaces, and special tokens.
    Example:
        "crazy.. available only in bugis" -> "crazy available only in bugis"
    """
    # First protect special tokens
    tokens = {}
    for i, token in enumerate(['<URL>', '<EMAIL>', '<PHONE>', '<NUM>']):
        placeholder = f'__TOKEN{i}__'
        tokens[placeholder] = token
        text = text.replace(token, placeholder)
    
    # Remove punctuation except for protected tokens
    text = re.sub(r'[^\w\s]', ' ', str(text))
    
    # Restore special tokens
    for placeholder, token in tokens.items():
        text = text.replace(placeholder, token)
    
    return text

def normalize_whitespace(text):
    """Collapse repeated spaces to single space and trim.
    Example:
        "go until jurong point  crazy   available" -> "go until jurong point crazy available"
    """
    return ' '.join(str(text).split())

def remove_stopwords(text):
    """Remove English stopwords when enabled.
    Example:
        "this is only a test of the system" -> "test system"
    """
    words = str(text).split()
    return ' '.join(w for w in words if w.lower() not in ENGLISH_STOP_WORDS)

def preprocess_text(text, remove_stopwords_flag=False, keep_numbers=False):
    """Apply full preprocessing pipeline in the correct order."""
    text = str(text)
    # Pipeline order matches the reference implementation
    text = lowercase_normalize(text)                      # 1. Lowercase
    text = mask_contacts(text)                           # 2. Mask contacts (URLs, emails, phones)
    text = handle_numbers(text, keep_numbers)            # 3. Handle numbers
    text = strip_punctuation(text)                       # 4. Strip punctuation
    text = normalize_whitespace(text)                    # 5. Normalize whitespace
    if remove_stopwords_flag:
        text = remove_stopwords(text)                    # 6. Optional: Remove stopwords
    
    return {
        'text_lower': lowercase_normalize(text),
        'text_contacts_masked': mask_contacts(text),
        'text_numbers': handle_numbers(text, keep_numbers),
        'text_stripped': strip_punctuation(text),
        'text_whitespace': normalize_whitespace(text),
        'text_stopwords_removed': remove_stopwords(text) if remove_stopwords_flag else text,
        'text_clean': text
    }