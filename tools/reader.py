import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


def validate_url(url: str) -> None:
    """Reject non-public and unsupported URLs."""

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are allowed")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid URL")

    addresses = socket.getaddrinfo(parsed.hostname, None)

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise ValueError("Private or non-public URL is not allowed")


def read_webpage(url: str) -> dict:
    """Download a webpage and extract readable text."""

    validate_url(url)

    with httpx.Client(
        timeout=10.0,
        follow_redirects=False,
    ) as client:

        with client.stream("GET", url) as response:
            response.raise_for_status()

            content_type = response.headers.get(
                "content-type", ""
            ).lower()

            if "text/html" not in content_type:
                raise ValueError("URL does not return HTML")

            chunks = []
            total_size = 0

            for chunk in response.iter_bytes():
                total_size += len(chunk)

                if total_size > 2_000_000:
                    raise ValueError("Webpage is too large")

                chunks.append(chunk)

            html = b"".join(chunks).decode(
                response.encoding or "utf-8",
                errors="replace",
            )

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script", "style", "nav", "footer",
        "header", "noscript",
    ]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    text = soup.get_text(separator=" ", strip=True)

    return {
        "url": url,
        "title": title,
        "content": text[:12000],
    }
