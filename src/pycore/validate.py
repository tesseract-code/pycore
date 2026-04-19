from typing import Optional, Sequence
from urllib.parse import urlparse


def validate_url(url: str,
                 allowed_schemes: Optional[Sequence[str]] = None,
                 require_tld: bool = True) -> tuple[bool, str]:
    """
    URL validation against allowed_schemes.

    Args:
        url: URL to validate
        allowed_schemes: List of allowed schemes (default: ['http', 'https'])
        require_tld: Whether to require a top-level domain

    Returns:
        tuple: (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"URL parsing failed: {e}"

    # Check scheme
    if not parsed.scheme:
        return False, "URL missing scheme (http/https)"

    if parsed.scheme not in allowed_schemes:
        return (False,
                f"Scheme '{parsed.scheme}' not allowed. Use: {allowed_schemes}")

    # Check netloc (domain)
    if not parsed.netloc:
        return False, "URL missing domain/host"

    # Check for TLD if required
    if require_tld and '.' not in parsed.netloc:
        return False, "URL missing top-level domain"

    return True, "Valid URL"
