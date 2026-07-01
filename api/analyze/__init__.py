import json
import logging
import os
import time
import urllib.error
import urllib.request

import azure.functions as func

# Read these from Azure environment variables / application settings —
# never hard-code the key here.
ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
TEXT_API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 20
MAX_CHARS = 5000


def main(req: func.HttpRequest) -> func.HttpResponse:
    endpoint = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("LANGUAGE_KEY", "")
    logging.info(f"Endpoint: {endpoint}")
    logging.info(f"Key exists: {bool(key)}")
    if not endpoint or not key:
        return _json_response(
            {"error": "Server is missing LANGUAGE_ENDPOINT / LANGUAGE_KEY environment variables."},
            500,
        )

    try:
        body = req.get_json()
    except ValueError:
        body = {}

    text = (body.get("text") or "").strip()
    if not text:
        return _json_response({"error": 'Request body must include non-empty "text".'}, 400)
    if len(text) > MAX_CHARS:
        return _json_response({"error": f"Text must be {MAX_CHARS} characters or fewer."}, 400)

    try:
        sentiment_doc = _call_language("SentimentAnalysis", text)["results"]["documents"][0]
        keyphrase_doc = _call_language("KeyPhraseExtraction", text)["results"]["documents"][0]
        entity_doc = _call_language("EntityRecognition", text)["results"]["documents"][0]
        language_doc = _call_language("LanguageDetection", text)["results"]["documents"][0]
        pii_doc = _call_language("PiiEntityRecognition", text)["results"]["documents"][0]
        summary_text = _call_extractive_summary(text)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        return _json_response({"error": f"Language service call failed: {e}"}, 502)

    result = {
        "sentiment": sentiment_doc["sentiment"],  # positive | negative | neutral | mixed
        "confidenceScores": sentiment_doc["confidenceScores"],
        "keyPhrases": keyphrase_doc.get("keyPhrases", []),
        "entities": [
            {"text": e["text"], "category": e["category"]}
            for e in entity_doc.get("entities", [])
        ],
        "languageDetection": {
            "name": language_doc.get("detectedLanguage", {}).get("name", "unknown"),
            "iso6391Name": language_doc.get("detectedLanguage", {}).get("iso6391Name", ""),
            "confidenceScore": language_doc.get("detectedLanguage", {}).get("confidenceScore", 0),
            "scriptName": language_doc.get("detectedLanguage", {}).get("scriptName", ""),
            "scriptIso15924Code": language_doc.get("detectedLanguage", {}).get("scriptIso15924Code", ""),
        },
        "pii": {
            "redactedText": pii_doc.get("redactedText", ""),
            "entities": [
                {
                    "text": e.get("text", ""),
                    "category": e.get("category", ""),
                }
                for e in pii_doc.get("entities", [])
            ],
        },
        "summary": {
            "extractive": summary_text,
        },
    }
    return _json_response(result, 200)


def _call_language(kind: str, text: str) -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"
    if kind == "LanguageDetection":
        document = {"id": "1", "countryHint": "us", "text": text}
    else:
        document = {"id": "1", "language": "en", "text": text}
    payload = {
        "kind": kind,
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {"documents": [document]},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{kind} failed: {exc.code} {detail}") from exc


def _call_extractive_summary(text: str) -> list[str]:
    url = f"{ENDPOINT}/language/analyze-text/jobs?api-version={TEXT_API_VERSION}"
    payload = {
        "displayName": "Review summary",
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": text,
                }
            ]
        },
        "tasks": [
            {
                "kind": "ExtractiveSummarization",
                "taskName": "Review summary task",
                "parameters": {
                    "sentenceCount": 3,
                },
            }
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            operation_location = resp.headers.get("operation-location") or resp.headers.get("Operation-Location")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"ExtractiveSummarization failed to start: {exc.code} {detail}") from exc

    if not operation_location:
        raise RuntimeError("ExtractiveSummarization request did not return an operation-location header.")

    status = None
    for _ in range(12):
        poll_request = urllib.request.Request(
            operation_location,
            headers={
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": KEY,
            },
            method="GET",
        )
        with urllib.request.urlopen(poll_request, timeout=TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        status = payload.get("status")
        if status == "succeeded":
            items = payload.get("tasks", {}).get("items", [])
            if not items:
                return []
            documents = items[0].get("results", {}).get("documents", [])
            if not documents:
                return []
            summaries = documents[0].get("summaries", [])
            return [summary.get("text", "") for summary in summaries if summary.get("text")]
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"ExtractiveSummarization job ended with status: {status}")
        time.sleep(0.5)

    raise RuntimeError(f"ExtractiveSummarization did not complete in time (last status: {status or 'unknown'}).")


def _json_response(payload: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )
