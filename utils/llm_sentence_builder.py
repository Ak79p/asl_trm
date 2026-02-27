import requests

HF_API_URL = "https://router.huggingface.co/models/google/flan-t5-base"


def build_sentence_from_windows(window_predictions, hf_token):
    """
    Convert ASL gloss into natural English using FLAN-T5 (free tier).
    """

    # ---- Extract top-1 word per window ----
    words = []

    for win in window_predictions:
        if win["words"]:
            words.append(win["words"][0][0].lower())

    # Remove consecutive duplicates
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)

    if not cleaned:
        return "No valid gloss extracted."

    gloss = " ".join(cleaned)

    prompt = (
        "Rewrite this ASL gloss as a natural English sentence:\n"
        f"{gloss}"
    )

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.3
        }
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return f"HF API Error: {response.text}"

    output = response.json()

    if isinstance(output, list) and "generated_text" in output[0]:
        return output[0]["generated_text"].strip()

    return str(output)