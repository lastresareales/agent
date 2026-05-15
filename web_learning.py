import re
from html import unescape
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from memory_store import save_message
from recognition_service import extract_text


MAX_PAGE_BYTES = 1_000_000
MAX_LEARN_CHARS = 5000


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self):
        return normalize_text(" ".join(self.parts))


def normalize_text(text):
    return re.sub(r"\s+", " ", unescape(text)).strip()


def validate_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be an http or https address")


def fetch_url_text(url):
    validate_url(url)
    request = Request(url, headers={"User-Agent": "EntityRecognitionMemoryBot/0.1"})
    try:
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(MAX_PAGE_BYTES + 1)
    except (OSError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not fetch URL: {error}") from error

    if len(body) > MAX_PAGE_BYTES:
        raise RuntimeError("Page is too large to ingest")

    decoded = body.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in decoded[:500].lower():
        parser = TextExtractor()
        parser.feed(decoded)
        return parser.text()

    return normalize_text(decoded)


def learn_url(url):
    text = fetch_url_text(url)
    selected_text = text[:MAX_LEARN_CHARS]
    save_message("system", selected_text, source=url, importance=0.7)
    extraction = extract_text(selected_text, learn=True)

    return {
        "url": url,
        "characters_read": len(text),
        "characters_learned": len(selected_text),
        "extraction": extraction,
    }
