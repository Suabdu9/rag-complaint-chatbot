import re


def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    boilerplate = [
        "i am writing to file a complaint",
        "i am writing to complain",
        "this complaint is about"
    ]

    for phrase in boilerplate:
        text = text.replace(phrase, "")

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()
