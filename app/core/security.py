"""Cryptographic security, HMAC token generation, and authentication verification."""

import hashlib
import hmac
import logging
import secrets
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

TOKEN_PREFIX = "srag_"


def get_active_secret_key() -> str:
    """Retrieve active SECRET_KEY from application settings.
    
    Raises
    ------
    RuntimeError
        If SECRET_KEY is not configured in settings.
    """
    settings = get_settings()
    key = settings.SECRET_KEY
    if not key or not key.strip():
        raise RuntimeError(
            "SECRET_KEY is not configured. Configure SECRET_KEY in environment or application settings."
        )
    return key.strip()


def get_active_admin_key() -> str:
    """Retrieve active ADMIN_API_KEY from application settings.
    
    Raises
    ------
    RuntimeError
        If ADMIN_API_KEY is not configured in settings.
    """
    settings = get_settings()
    key = settings.ADMIN_API_KEY
    if not key or not key.strip():
        raise RuntimeError(
            "ADMIN_API_KEY is not configured. Configure ADMIN_API_KEY in environment or application settings."
        )
    return key.strip()


def create_tenant_token(tenant_id: str, secret_key: Optional[str] = None) -> str:
    """Generate a tamper-proof HMAC-SHA256 authenticated tenant token.

    Parameters
    ----------
    tenant_id : str
        Sanitized identifier of the tenant organization.
    secret_key : str, optional
        Secret signing key. Defaults to settings.SECRET_KEY.

    Returns
    -------
    str
        Bearer token string formatted as 'srag_<tenant_id>_<hex_signature>'.
    """
    clean_tenant = tenant_id.strip()
    if not clean_tenant:
        raise ValueError("tenant_id cannot be empty.")

    key = secret_key or get_active_secret_key()
    sig = hmac.new(
        key.encode("utf-8"),
        f"tenant:{clean_tenant}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]
    return f"{TOKEN_PREFIX}{clean_tenant}_{sig}"


def verify_tenant_token(token: str, secret_key: Optional[str] = None) -> Optional[str]:
    """Verify an authenticated tenant token and extract its verified tenant_id.

    Parameters
    ----------
    token : str
        Incoming token string.
    secret_key : str, optional
        Secret signing key. Defaults to settings.SECRET_KEY.

    Returns
    -------
    Optional[str]
        Verified tenant_id if signature is valid; None if forged or invalid.
    """
    if not token or not token.startswith(TOKEN_PREFIX):
        return None

    body = token[len(TOKEN_PREFIX):]
    parts = body.rsplit("_", 1)
    if len(parts) != 2:
        return None

    tenant_id, signature = parts
    if not tenant_id or not signature:
        return None

    try:
        key = secret_key or get_active_secret_key()
    except RuntimeError:
        return None

    expected_sig = hmac.new(
        key.encode("utf-8"),
        f"tenant:{tenant_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]

    if hmac.compare_digest(signature, expected_sig):
        return tenant_id
    return None


def verify_admin_key(key: str, admin_api_key: Optional[str] = None) -> bool:
    """Constant-time validation of administrative credentials.

    Parameters
    ----------
    key : str
        Incoming admin key candidate.
    admin_api_key : str, optional
        Authorized admin key. Defaults to settings.ADMIN_API_KEY.

    Returns
    -------
    bool
        True if key matches active ADMIN_API_KEY, False otherwise.
    """
    if not key:
        return False
    try:
        authorized = admin_api_key or get_active_admin_key()
    except RuntimeError:
        return False
    return hmac.compare_digest(key.strip(), authorized.strip())
