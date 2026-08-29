"""
core/agent/scraper.py — Autonomous PDF Discovery Engine

Given a portal/listing URL supplied by the user, discovers the target
certificate/compliance PDF URLs without drifting into unrelated links.

Pipeline:
  fetch -> extract links -> Phase A deterministic guards (same-host scoping,
  keyword block/allow lists, PDF hints, crawl budget) -> Phase B agentic LLM
  selection -> hard verification (Content-Type: application/pdf + %PDF magic
  bytes) -> dedup against the database and the local fetched-URL manifest.

Local-first: uses Python stdlib html.parser + requests. Optional headless Chromium
via Playwright for JS-rendered SPAs, and optional transient session-cookie auth
for portal-login deployments (cookie is held in memory only, never stored/logged).
"""

import html.parser
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

#: PDF magic bytes used for hard content verification.
_PDF_MAGIC = b"%PDF"

#: URL heuristics that indicate a likely document download.
_PDF_HINT_RE = re.compile(
    r"(\.pdf$|/download/|/file/|/document|/documents|/getdocument|/attachment"
    r"|documentview|\.aspx\?|[?&]id=|[?&]docid=|[?&]download=|[?&]attachment=)",
    re.IGNORECASE,
)

#: Navigation / non-document page keywords (Phase A blocklist).
DEFAULT_BLOCKED_KEYWORDS: Tuple[str, ...] = (
    "login", "logout", "signin", "signup", "register", "password", "account",
    "privacy", "terms", "contact", "about", "sitemap", "help", "faq", "career",
    "news", "press", "javascript:", "mailto:", "tel:",
)

#: Target document keywords (anchor text / URL hints for certificate docs).
DEFAULT_ALLOWED_KEYWORDS: Tuple[str, ...] = (
    "certificate", "certif", "homologat", "homologation", "compliance",
    "conformity", "approval", "attestation", "document", "pdf", "download",
)


@dataclass
class DiscoveredLink:
    url: str
    source_page: str
    anchor_text: str = ""
    content_type: Optional[str] = None
    is_pdf_hint: bool = False


@dataclass
class DiscoveryResult:
    base_url: str = ""
    new_urls: List[str] = field(default_factory=list)      # LLM/heuristic-selected, pre-verification
    verified_urls: List[str] = field(default_factory=list)  # confirmed real PDFs, not in DB
    skipped_existing: List[str] = field(default_factory=list)  # already in DB / manifest
    excluded: List[str] = field(default_factory=list)          # Phase A rejects
    failed_verification: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "new_urls": self.new_urls,
            "verified_urls": self.verified_urls,
            "skipped_existing": self.skipped_existing,
            "excluded": self.excluded,
            "failed_verification": self.failed_verification,
            "reason": self.reason,
        }


class HtmlFetcher:
    """Fetches an HTML page with requests. Returns (html, final_url)."""

    def __init__(
        self,
        timeout: int = 30,
        user_agent: str = "RFComplianceBot/1.0",
        polite_delay: float = 0.0,
        cookie_header: Optional[str] = None,
    ) -> None:
        self.timeout = timeout
        self.polite_delay = polite_delay
        self.cookie_header = cookie_header
        self.session = requests.Session()
        headers = {"User-Agent": user_agent}
        if cookie_header:
            # Transient session cookie: used in-memory only, never stored/logged.
            headers["Cookie"] = cookie_header
        self.session.headers.update(headers)

    def fetch(self, url: str) -> Tuple[str, str]:
        if self.polite_delay:
            time.sleep(self.polite_delay)
        resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        final_url = resp.url
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype.lower():
            return "", final_url
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text, final_url

    def close(self) -> None:
        """No-op for the requests fetcher (no resources to release)."""


class PlaywrightFetcher:
    """
    Headless-browser fetcher for JS-rendered SPAs (e.g. OpenText/SharePoint
    document portals). Launches a single Chromium instance lazily and reuses it
    across fetches within a discovery run; call close() when done.

    Optional transient session cookie is applied via context headers and is never
    stored or logged.
    """

    def __init__(
        self,
        timeout: int = 30,
        user_agent: str = "RFComplianceBot/1.0",
        polite_delay: float = 0.0,
        cookie_header: Optional[str] = None,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.polite_delay = polite_delay
        self.cookie_header = cookie_header
        self._pw = None
        self._browser = None

    def _get_browser(self):
        if self._browser is None:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True, args=["--no-sandbox"])
        return self._browser

    def fetch(self, url: str) -> Tuple[str, str]:
        if self.polite_delay:
            time.sleep(self.polite_delay)
        try:
            browser = self._get_browser()
            context = browser.new_context(user_agent=self.user_agent)
            if self.cookie_header:
                context.set_extra_http_headers({"Cookie": self.cookie_header})
            page = context.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
            except Exception:
                # Some SPAs never reach network idle (polling); fall back to a
                # settled render with a short post-load wait.
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                    page.wait_for_timeout(2000)
                except Exception as exc:
                    logger.info(f" Playwright fetch failed for {url}: {exc}")
                    context.close()
                    return "", url
            html = page.content()
            final_url = page.url
            context.close()
            return html, final_url
        except Exception as exc:
            logger.info(f" Playwright fetch failed for {url}: {exc}")
            return "", url

    def close(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
            if self._pw is not None:
                self._pw.stop()
        except Exception as exc:
            logger.warning(f" Playwright fetcher close error: {exc}")
        self._browser = None
        self._pw = None


def create_fetcher(fetcher_type: str = "html", **kwargs) -> Any:
    """
    Factory for the configured fetcher backend.

    Args:
        fetcher_type: "html" (requests, default) or "playwright" (headless browser
            for JS-rendered SPAs; requires the playwright package + chromium).
        **kwargs: forwarded to the fetcher constructor (timeout, user_agent,
            polite_delay, cookie_header).

    Raises:
        ImportError: playwright requested but not installed.
    """
    fetcher_type = str(fetcher_type or "html").strip().lower()
    if fetcher_type == "playwright":
        try:
            import playwright  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SCRAPER_FETCHER=playwright requires the 'playwright' package "
                "and `playwright install chromium`. Install them to use browser rendering."
            ) from exc
        return PlaywrightFetcher(**kwargs)
    return HtmlFetcher(**kwargs)


class _LinkExtractor(html.parser.HTMLParser):
    """Collects (href, anchor_text) pairs using only the stdlib HTML parser."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._anchor: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag in ("a", "iframe", "frame"):
            href = attrs.get("href") or attrs.get("src")
            if href:
                self._current_href = href.strip()
                self._anchor = []
        elif tag == "form":
            action = attrs.get("action")
            if action:
                self.links.append((action.strip(), ""))

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._anchor.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("a", "iframe", "frame") and self._current_href is not None:
            text = " ".join("".join(self._anchor).split())
            self.links.append((self._current_href, text[:200]))
            self._current_href = None
            self._anchor = []


def _same_host(url: str, base_url: str) -> bool:
    """True if url's host equals the base host or is a subdomain of it."""
    u = urlparse(url)
    b = urlparse(base_url)
    if not u.hostname or not b.hostname:
        return False
    return u.hostname == b.hostname or u.hostname.endswith("." + b.hostname)


def extract_links(html: str, base_url: str) -> List[DiscoveredLink]:
    """Parse raw HTML into absolute DiscoveredLink objects (deduplicated)."""
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception as exc:
        logger.warning(f" HTML parse error on {base_url}: {exc}")

    seen: Set[str] = set()
    links: List[DiscoveredLink] = []
    for href, anchor in parser.links:
        href = href.strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        abs_url = urljoin(base_url, href)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append(
            DiscoveredLink(
                url=abs_url,
                source_page=base_url,
                anchor_text=anchor,
                is_pdf_hint=bool(_PDF_HINT_RE.search(abs_url)),
            )
        )
    return links


def _contains_keyword(text: str, keyword: str) -> bool:
    """Whole-word keyword match (prevents e.g. 'press' matching 'espressif')."""
    try:
        return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None
    except re.error:
        return keyword in text


def filter_candidates(
    links: List[DiscoveredLink],
    base_url: str,
    allowed_keywords: Tuple[str, ...] = DEFAULT_ALLOWED_KEYWORDS,
    blocked_keywords: Tuple[str, ...] = DEFAULT_BLOCKED_KEYWORDS,
) -> Tuple[List[DiscoveredLink], List[str]]:
    """
    Phase A: same-host scoping + blocklist + PDF/document hints.
    Returns (kept, excluded_urls).
    """
    kept: List[DiscoveredLink] = []
    excluded: List[str] = []
    for link in links:
        if not _same_host(link.url, base_url):
            excluded.append(link.url)
            continue
        lower = f"{link.url} {link.anchor_text}".lower()
        if any(_contains_keyword(lower, kw) for kw in blocked_keywords):
            excluded.append(link.url)
            continue
        if link.is_pdf_hint or any(_contains_keyword(lower, kw) for kw in allowed_keywords):
            kept.append(link)
        else:
            excluded.append(link.url)
    return kept, excluded


def filter_pdf_links_with_llm(
    links: List[DiscoveredLink],
    base_url: str,
) -> Tuple[List[str], str]:
    """
    Phase B: ask the local LLM to select only the target document links.

    Returns (selected_urls, reason). On LLM failure, falls back to the PDF-hint
    candidates to bound drift.
    """
    from core.prompts import PDF_LINK_SELECTION_SYSTEM_PROMPT
    from core.llm import generate_json

    if not links:
        return [], "no candidates"

    lines = [f"{i}. URL: {link.url}\n   Anchor: {link.anchor_text or '(none)'}" for i, link in enumerate(links, 1)]
    user_prompt = (
        f"PORTAL/BASE URL: {base_url}\n\n"
        f"CANDIDATE LINKS:\n" + "\n".join(lines) + "\n\n"
        f"Return ONLY the raw JSON output matching the schema."
    )
    try:
        raw_response = generate_json(
            system_prompt=PDF_LINK_SELECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            disable_thinking=True,
        )
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            selected = [str(u).strip() for u in parsed.get("selected_urls", [])]
            valid = {link.url for link in links}
            selected = [u for u in selected if u in valid]
            reason = str(parsed.get("reason", "")).strip() or "LLM-selected target documents."
            if selected:
                return selected, reason
            # Empty LLM selection is NOT trusted when obvious PDF-hint candidates
            # exist (a single flaky verdict would otherwise drop real documents).
            # The hard PDF verification below is the authoritative gate.
            pdf_hints = [link.url for link in links if link.is_pdf_hint]
            if pdf_hints:
                logger.warning(
                    " LLM returned an empty selection despite PDF-hint candidates; "
                    "using them (hard PDF verification will filter)."
                )
                return pdf_hints, "LLM returned an empty selection; fell back to PDF-hint candidates."
    except Exception as exc:
        logger.warning(f" LLM link selection failed ({exc}); using heuristic PDF-hint fallback.")

    fallback = [link.url for link in links if link.is_pdf_hint]
    if not fallback:
        fallback = [link.url for link in links]
    return fallback, "LLM selection unavailable; heuristic fallback used."


def verify_pdf(
    url: str,
    timeout: int = 30,
    user_agent: str = "RFComplianceBot/1.0",
    cookie_header: Optional[str] = None,
) -> Optional[str]:
    """
    Confirm the URL serves a real PDF (Content-Type application/pdf OR %PDF magic
    bytes). Returns the content-type string on success, else None.
    """
    headers = {"User-Agent": user_agent}
    if cookie_header:
        headers["Cookie"] = cookie_header
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True, headers=headers) as resp:
            resp.raise_for_status()
            head = next(resp.iter_content(5), b"")
            ctype = resp.headers.get("Content-Type", "").lower()
            if head.startswith(_PDF_MAGIC) or "pdf" in ctype:
                return ctype or "application/pdf"
    except Exception as exc:
        logger.info(f" PDF verification failed for {url}: {exc}")
    return None


def load_fetched_manifest(manifest_path: Optional[str]) -> Set[str]:
    """Load the append-only manifest of already-fetched URLs (lowercased)."""
    if not manifest_path:
        return set()
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            return {str(u).strip().lower() for u in json.load(fh)}
    except Exception:
        return set()


def append_manifest(urls: List[str], manifest_path: Optional[str]) -> None:
    """Append URLs to the fetched manifest (idempotent)."""
    if not manifest_path:
        return
    current = load_fetched_manifest(manifest_path)
    current.update(str(u).strip().lower() for u in urls)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(sorted(current), fh, indent=2)


def existing_db_urls(db_session) -> Set[str]:
    """Return the set of certificate cert_link values already persisted (lowercased)."""
    try:
        from schemas.extraction import CertificateMetadata
        rows = (
            db_session.query(CertificateMetadata.cert_link)
            .filter(CertificateMetadata.cert_link.isnot(None))
            .all()
        )
        return {str(r[0]).strip().lower() for r in rows if r[0]}
    except Exception as exc:
        logger.warning(f" DB cert_link dedup check failed: {exc}")
        return set()


#: RF / radio-telecom signal patterns (content gate).
_RF_SIGNAL_RE = re.compile(
    r"\b(radio[-\s]?frequency|\brf\b|rf module|rf transmitter|telecom(?:munication)?|"
    r"wireless|\bgsm\b|\bumts\b|\blte\b|\b5g\b|\b4g\b|\b3g\b|wi-?fi|bluetooth|"
    r"frequency|\bmhz\b|\bghz\b|antenna|transmitter|receiver|\beirp\b|\bsar\b|"
    r"band\b)\b",
    re.IGNORECASE,
)

#: Certificate / homologation signal patterns (content gate).
_CERT_SIGNAL_RE = re.compile(
    r"\b(certificate|certificado|certificat|certification|homologation|homologa|"
    r"approval|attestation|conformity|compliance|declaration of conformity|"
    r"registro|disposici[oó]n|resoluci[oó]n)\b",
    re.IGNORECASE,
)


def _extract_pdf_text(file_path: str, max_chars: int = 6000) -> str:
    """
    Extract embedded text from a PDF file (best-effort, pypdf).

    Scanned PDFs have no embedded text and return "" — the caller classifies
    those as "unclear" (kept for ingest verification). Only pypdf extraction is
    used: raw-byte scanning is intentionally avoided because PDF structural noise
    (obj/endobj/stream...) would misclassify scanned documents as irrelevant.
    """
    text = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        for page in reader.pages[:5]:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text += page_text
            if len(text) >= max_chars:
                break
    except Exception as exc:
        logger.debug(f" pypdf text extraction failed for {file_path}: {exc}")

    return text[:max_chars]


def classify_pdf_relevance(file_path: str) -> Dict[str, Any]:
    """
    Classify a downloaded PDF as relevant to RF certificates based on embedded
    content.

    Returns:
        Dict with:
          - status: "relevant" (certificate + RF signals found),
                    "unclear" (no extractable text - scanned; keep for ingest
                    verification), or "irrelevant" (text present but not an RF
                    certificate).
          - signals: {"certificate": bool, "rf": bool, "text_chars": int}
    """
    text = _extract_pdf_text(file_path)
    signals = {
        "certificate": bool(_CERT_SIGNAL_RE.search(text)),
        "rf": bool(_RF_SIGNAL_RE.search(text)),
        "text_chars": len(text.strip()),
    }
    if not text.strip():
        status = "unclear"  # scanned / no embedded text; the ingest OCR is the gate
    elif signals["certificate"] and signals["rf"]:
        status = "relevant"
    else:
        status = "irrelevant"
    logger.info(
        f" Relevance({file_path}): {status} "
        f"(cert={signals['certificate']}, rf={signals['rf']}, text_chars={signals['text_chars']})"
    )
    return {"status": status, "signals": signals}


def _crawl_candidates(
    base_url: str,
    fetcher: Any,
    *,
    max_pages: int,
    max_depth: int,
    allowed_keywords: Tuple[str, ...],
    blocked_keywords: Tuple[str, ...],
) -> Tuple[List[DiscoveredLink], List[str]]:
    """Breadth-first crawl collecting candidate links + excluded URLs."""
    queue: List[Tuple[str, int]] = [(base_url, 0)]
    visited: Set[str] = set()
    candidates: List[DiscoveredLink] = []
    excluded: List[str] = []
    pages_visited = 0

    while queue and pages_visited < max_pages:
        page_url, depth = queue.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)
        try:
            html, _final = fetcher.fetch(page_url)
        except Exception as exc:
            logger.info(f" Fetch failed for {page_url}: {exc}")
            continue
        if not html:
            continue
        pages_visited += 1
        links = extract_links(html, page_url)
        kept, excl = filter_candidates(
            links, base_url, allowed_keywords=allowed_keywords, blocked_keywords=blocked_keywords
        )
        candidates.extend(kept)
        excluded.extend(excl)

        if depth < max_depth:
            for link in links:
                if link.is_pdf_hint:
                    continue
                if _same_host(link.url, base_url) and link.url not in visited:
                    queue.append((link.url, depth + 1))
    return candidates, excluded


def _finalize_result(
    result: DiscoveryResult,
    candidates: List[DiscoveredLink],
    *,
    base_url: str,
    already: Set[str],
    use_llm: bool,
    timeout: int,
    user_agent: str,
    cookie_header: Optional[str],
) -> List[str]:
    """Run dedup -> LLM selection -> hard PDF verification; fills result fields and
    returns the verified URL list."""
    fresh = [link for link in candidates if link.url.lower() not in already]
    result.skipped_existing = [link.url for link in candidates if link.url.lower() in already]

    if use_llm and fresh:
        selected, reason = filter_pdf_links_with_llm(fresh, base_url=base_url)
    else:
        selected = [link.url for link in fresh]
        reason = "LLM filter disabled; heuristic candidates used."
    result.new_urls = selected
    result.reason = reason

    verified: List[str] = []
    failed: List[str] = []
    for url in selected:
        ctype = verify_pdf(url, timeout=timeout, user_agent=user_agent, cookie_header=cookie_header)
        if ctype:
            verified.append(url)
        else:
            failed.append(url)
    result.verified_urls = verified
    result.failed_verification = failed
    return verified


def discover_pdf_urls(
    base_url: str,
    *,
    db_session=None,
    max_pages: int = 10,
    max_depth: int = 2,
    timeout: int = 30,
    user_agent: str = "RFComplianceBot/1.0",
    polite_delay: float = 0.5,
    allowed_keywords: Tuple[str, ...] = DEFAULT_ALLOWED_KEYWORDS,
    blocked_keywords: Tuple[str, ...] = DEFAULT_BLOCKED_KEYWORDS,
    manifest_path: Optional[str] = None,
    use_llm: bool = True,
    fetcher_type: str = "auto",
    cookie_header: Optional[str] = None,
) -> DiscoveryResult:
    """
    Discover target PDF URLs reachable from a portal/listing URL.

    Args:
        base_url: user-supplied portal URL (e.g. a document database index page).
        db_session: optional SQLAlchemy session for cert_link dedup.
        max_pages / max_depth: crawl budget.
        manifest_path: local manifest of already-fetched URLs (dedup).
        use_llm: run Phase B agentic selection.
        fetcher_type: "auto" (default) tries the fast html fetcher first and
            transparently retries with playwright when the html crawl yields no
            verified PDFs (i.e. the portal is JS-rendered - even when its raw HTML
            shell exposes non-PDF candidate links); "html" or "playwright" force a
            specific backend.
        cookie_header: optional transient session cookie header value applied to
            HTML fetches and PDF verification (never stored or logged).

    Returns:
        DiscoveryResult with verified (new) PDF URLs, skipped-existing, excluded.
    """
    result = DiscoveryResult(base_url=base_url)
    already = load_fetched_manifest(manifest_path)
    if db_session is not None:
        already |= existing_db_urls(db_session)
    already = {u.lower() for u in already}

    used_fetcher = str(fetcher_type or "auto").lower()

    if used_fetcher == "auto":
        # Fast path first: server-rendered portals are fetched with requests.
        html_fetcher = create_fetcher(
            "html", timeout=timeout, user_agent=user_agent,
            polite_delay=polite_delay, cookie_header=cookie_header,
        )
        try:
            candidates, excluded = _crawl_candidates(
                base_url, html_fetcher,
                max_pages=max_pages, max_depth=max_depth,
                allowed_keywords=allowed_keywords, blocked_keywords=blocked_keywords,
            )
        finally:
            html_fetcher.close()
        used_fetcher = "html"
        result.excluded = excluded
        verified = _finalize_result(
            result, candidates,
            base_url=base_url, already=already, use_llm=use_llm,
            timeout=timeout, user_agent=user_agent, cookie_header=cookie_header,
        )

        if not verified:
            # The html crawl produced no real PDFs (JS-rendered SPA: its raw shell
            # exposes nav URLs that fail PDF verification). Retry with headless
            # Chromium automatically.
            logger.info(" HTML crawl produced no verified PDFs; retrying with Playwright (auto).")
            try:
                pw_fetcher = create_fetcher(
                    "playwright", timeout=timeout, user_agent=user_agent,
                    polite_delay=polite_delay, cookie_header=cookie_header,
                )
                try:
                    candidates, excluded = _crawl_candidates(
                        base_url, pw_fetcher,
                        max_pages=max_pages, max_depth=max_depth,
                        allowed_keywords=allowed_keywords, blocked_keywords=blocked_keywords,
                    )
                finally:
                    pw_fetcher.close()
                used_fetcher = "playwright"
                result.excluded = excluded
                verified = _finalize_result(
                    result, candidates,
                    base_url=base_url, already=already, use_llm=use_llm,
                    timeout=timeout, user_agent=user_agent, cookie_header=cookie_header,
                )
            except ImportError as exc:
                logger.warning(f" Playwright fallback unavailable (auto): {exc}")
    else:
        fetcher = create_fetcher(
            used_fetcher, timeout=timeout, user_agent=user_agent,
            polite_delay=polite_delay, cookie_header=cookie_header,
        )
        try:
            candidates, excluded = _crawl_candidates(
                base_url, fetcher,
                max_pages=max_pages, max_depth=max_depth,
                allowed_keywords=allowed_keywords, blocked_keywords=blocked_keywords,
            )
        finally:
            fetcher.close()
        result.excluded = excluded
        verified = _finalize_result(
            result, candidates,
            base_url=base_url, already=already, use_llm=use_llm,
            timeout=timeout, user_agent=user_agent, cookie_header=cookie_header,
        )

    logger.info(
        f" Discovery({base_url}, fetcher={used_fetcher}): {len(verified)} verified new PDF(s), "
        f"{len(result.skipped_existing)} already fetched, {len(result.failed_verification)} failed verification."
    )
    return result