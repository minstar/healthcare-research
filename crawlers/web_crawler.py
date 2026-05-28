"""General web crawler for medical open questions from diverse sources.

Targets: WHO priority documents, StackExchange Medical Sciences,
Wikipedia medical open questions, and general web search results.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterator
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .base import BaseCrawler, RawDocument

logger = logging.getLogger(__name__)

SE_API = "https://api.stackexchange.com/2.3"

WHO_PRIORITY_URLS = [
    "https://www.who.int/activities/prioritizing-diseases-for-research-and-development-in-emergency-contexts",
]


class WebCrawler(BaseCrawler):
    """Crawls miscellaneous web sources for open medical questions."""

    def __init__(self, output_dir, rate_limit_sec: float = 2.0):
        super().__init__("web", output_dir, rate_limit_sec)

    def _crawl_stackexchange(self, sites: list[str], max_results: int) -> Iterator[RawDocument]:
        per_site = max(max_results // len(sites), 50)
        tags_medical = [
            "unanswered", "open-question", "research",
            "diagnosis", "treatment", "pathology",
        ]

        for site in sites:
            logger.info(f"[web/SE] Crawling {site}")
            params = {
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
                "pagesize": min(per_site, 100),
                "site": site,
            }

            # Unanswered questions with high votes = likely open questions
            unanswered_url = f"{SE_API}/questions/unanswered"
            try:
                resp = requests.get(unanswered_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"[web/SE] {site} failed: {e}")
                continue

            for item in data.get("items", []):
                title = item.get("title", "")
                body = item.get("body", "")
                body_text = BeautifulSoup(body, "html.parser").get_text()
                tags = item.get("tags", [])

                yield RawDocument(
                    source=f"stackexchange/{site}",
                    source_id=str(item.get("question_id", "")),
                    url=item.get("link", ""),
                    title=title,
                    abstract=body_text[:2000],
                    document_type="forum_question",
                    keywords=tags,
                    metadata={
                        "score": item.get("score", 0),
                        "view_count": item.get("view_count", 0),
                        "answer_count": item.get("answer_count", 0),
                    },
                )

            self.rate_limit()

    def _crawl_who_priorities(self) -> Iterator[RawDocument]:
        headers = {
            "User-Agent": "ResearchMed-Crawler/1.0 (medical research; minstar@upstage.ai)"
        }
        for url in WHO_PRIORITY_URLS:
            logger.info(f"[web/WHO] Fetching {url}")
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                title = soup.find("title")
                title_text = title.get_text().strip() if title else "WHO Priority Document"

                main = soup.find("main") or soup.find("article") or soup.find("body")
                if main:
                    text = main.get_text(separator="\n", strip=True)
                else:
                    text = soup.get_text(separator="\n", strip=True)

                yield RawDocument(
                    source="who",
                    source_id=urlparse(url).path.strip("/").replace("/", "_"),
                    url=url,
                    title=title_text,
                    abstract=text[:3000],
                    full_text=text,
                    document_type="priority_document",
                    metadata={"organization": "WHO"},
                )
            except Exception as e:
                logger.error(f"[web/WHO] Failed to fetch {url}: {e}")

            self.rate_limit()

    def _crawl_wikipedia_open_problems(self) -> Iterator[RawDocument]:
        wiki_api = "https://en.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "ResearchMed-Crawler/1.0 (medical research; minstar@upstage.ai)"
        }
        search_titles = [
            "List of unsolved problems in medicine",
            "List of unsolved problems in biology",
            "List of unsolved problems in neuroscience",
            "Open problems in bioinformatics",
        ]

        for title in search_titles:
            params = {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
                "format": "json",
            }
            try:
                resp = requests.get(wiki_api, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                pages = resp.json().get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    if page_id == "-1":
                        continue
                    yield RawDocument(
                        source="wikipedia",
                        source_id=f"wiki:{page_id}",
                        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        title=page.get("title", title),
                        abstract=page.get("extract", "")[:3000],
                        full_text=page.get("extract", ""),
                        document_type="encyclopedia",
                        metadata={"page_id": page_id},
                    )
            except Exception as e:
                logger.error(f"[web/wiki] Failed: {title}: {e}")

            self.rate_limit()

    def crawl(self, queries: list[str], max_results: int) -> Iterator[RawDocument]:
        yield from self._crawl_wikipedia_open_problems()
        yield from self._crawl_who_priorities()
        yield from self._crawl_stackexchange(
            sites=["medicalsciences", "biology", "bioinformatics"],
            max_results=max_results,
        )
