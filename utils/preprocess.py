import re


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"http\S+", "", text)            # remove urls
    # keep basic alnum and apostrophe
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    return text.strip().lower()
