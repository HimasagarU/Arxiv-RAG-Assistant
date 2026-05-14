"""
auth.py — JWT authentication endpoints and middleware.

Endpoints:
    POST /auth/register  — Create account, return tokens
    POST /auth/login     — Verify credentials, return tokens
    POST /auth/refresh   — Refresh access token
    GET  /auth/me        — Return current user profile

Security:
    - Passwords hashed with bcrypt (direct, no passlib)
    - JWTs signed with HS256 (python-jose)
    - Access token: 30 min | Refresh token: 7 days
    - Login rate limiting: 5 attempts/min per IP
"""

import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.app_database import get_app_db
from db.app_models import User

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
JWT_TOKEN_VERSION = int(os.getenv("JWT_TOKEN_VERSION", "1"))

_redis_state = None  # None=lazy, "disabled"=unavailable, else Redis client


def _sync_redis():
    global _redis_state
    if _redis_state == "disabled":
        return None
    if _redis_state is not None:
        return _redis_state
    url = (os.getenv("REDIS_URL", "") or "").strip()
    if not url:
        _redis_state = "disabled"
        return None
    try:
        import redis as sync_redis

        _redis_state = sync_redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
    except Exception as e:
        log.warning("Auth: Redis unavailable (%s); using in-memory rate limits only.", e)
        _redis_state = "disabled"
    return _redis_state if _redis_state != "disabled" else None


def _bcrypt_password_bytes(password: str, *, strict: bool) -> bytes:
    """Encode a password for bcrypt, optionally rejecting inputs over 72 bytes.

    bcrypt only uses the first 72 bytes of input. For new passwords we reject
    longer inputs so users do not accidentally create a password that is
    silently truncated. For verification we keep compatibility with existing
    stored hashes.
    """
    pwd_bytes = password.encode("utf-8")
    if strict and len(pwd_bytes) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be 72 bytes or fewer after UTF-8 encoding.",
        )
    return pwd_bytes[:72]


def hash_password(password: str) -> str:
    """Hash a password using bcrypt directly."""
    pwd_bytes = _bcrypt_password_bytes(password, strict=True)
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    pwd_bytes = _bcrypt_password_bytes(plain_password, strict=False)
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pwd_bytes, hashed_bytes)
security_scheme = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per IP)
# ---------------------------------------------------------------------------

_login_attempts: dict[str, list[float]] = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 60
MAX_EMAIL_FAILS = 8
EMAIL_LOCKOUT_SECONDS = 900


def _check_rate_limit(ip: str) -> None:
    """Raise 429 if IP has exceeded login attempt limit (Redis or in-memory)."""
    r = _sync_redis()
    now = time.time()
    if r:
        try:
            k = f"login:ip:{ip}"
            n = int(r.incr(k))
            if n == 1:
                r.expire(k, LOGIN_WINDOW_SECONDS)
            if n > MAX_LOGIN_ATTEMPTS:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Please wait a minute.",
                )
            return
        except HTTPException:
            raise
        except Exception as e:
            log.debug("Redis IP rate limit fallback: %s", e)

    _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < LOGIN_WINDOW_SECONDS]
    if len(_login_attempts[ip]) >= MAX_LOGIN_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please wait a minute.",
        )
    _login_attempts[ip].append(now)


def _check_email_lockout(email: str) -> None:
    r = _sync_redis()
    if r:
        try:
            if r.get(f"login:lock:{email.lower()}"):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed attempts for this account. Try again later.",
                )
        except HTTPException:
            raise
        except Exception:
            pass


def _record_login_failure(email: str) -> None:
    r = _sync_redis()
    key = email.lower()
    if r:
        try:
            fk = f"login:fail:{key}"
            n = int(r.incr(fk))
            if n == 1:
                r.expire(fk, EMAIL_LOCKOUT_SECONDS)
            if n >= MAX_EMAIL_FAILS:
                r.setex(f"login:lock:{key}", EMAIL_LOCKOUT_SECONDS, "1")
        except Exception as e:
            log.debug("login fail counter: %s", e)
    else:
        _email_fail_times[key].append(time.time())


_email_fail_times: dict[str, list[float]] = defaultdict(list)


def _clear_login_failures(email: str) -> None:
    r = _sync_redis()
    if r:
        try:
            r.delete(f"login:fail:{email.lower()}")
        except Exception:
            pass
    _email_fail_times.pop(email.lower(), None)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str = Field(default="Researcher", max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class UserProfile(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: str


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE)
    )
    to_encode.update(
        {"exp": expire, "type": "access", "ver": JWT_TOKEN_VERSION}
    )
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE)
    jti = str(uuid.uuid4())
    to_encode.update(
        {"exp": expire, "type": "refresh", "jti": jti, "ver": JWT_TOKEN_VERSION}
    )
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _refresh_revoked(jti: str) -> bool:
    if not jti:
        return False
    r = _sync_redis()
    if not r:
        return False
    try:
        return bool(r.exists(f"auth:revoked_refresh:{jti}"))
    except Exception:
        return False


def _revoke_refresh_jti(jti: str) -> None:
    if not jti:
        return
    r = _sync_redis()
    if not r:
        return
    try:
        r.setex(
            f"auth:revoked_refresh:{jti}",
            REFRESH_TOKEN_EXPIRE * 86400 + 60,
            "1",
        )
    except Exception as e:
        log.debug("revoke refresh: %s", e)


def _user_to_dict(user: User) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "created_at": user.created_at.isoformat() if user.created_at else "",
    }


# ---------------------------------------------------------------------------
# JWT dependency — extracts current user from Authorization header
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_app_db),
) -> User:
    """FastAPI dependency: validates JWT and returns the User ORM object."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type.",
            )
        if int(payload.get("ver", 1)) != JWT_TOKEN_VERSION:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token version invalid; please sign in again.",
            )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload.",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_app_db)):
    """Create a new user account and return JWT tokens."""
    # Check if email already registered
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered.",
        )

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        display_name=body.display_name,
    )
    db.add(user)
    await db.flush()  # get user.id before commit

    access = create_access_token({"sub": str(user.id)})
    refresh = create_refresh_token({"sub": str(user.id)})

    log.info(f"New user registered: {user.email}")
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        user=_user_to_dict(user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_app_db),
):
    """Authenticate user and return JWT tokens."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    _check_email_lockout(body.email)

    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.hashed_password):
        _record_login_failure(body.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    _clear_login_failures(body.email)
    access = create_access_token({"sub": str(user.id)})
    refresh = create_refresh_token({"sub": str(user.id)})

    log.info(f"User logged in: {user.email}")
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        user=_user_to_dict(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_app_db),
):
    """Exchange a refresh token for a new access + refresh token pair."""
    try:
        payload = jwt.decode(body.refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type.")
        if int(payload.get("ver", 1)) != JWT_TOKEN_VERSION:
            raise HTTPException(status_code=401, detail="Token version invalid.")
        user_id = payload.get("sub")
        jti = payload.get("jti")
        if _refresh_revoked(str(jti or "")):
            raise HTTPException(status_code=401, detail="Refresh token revoked.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Refresh token expired or invalid.")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found.")

    _revoke_refresh_jti(str(jti or ""))

    access = create_access_token({"sub": str(user.id)})
    refresh = create_refresh_token({"sub": str(user.id)})
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        user=_user_to_dict(user),
    )


@router.get("/me", response_model=UserProfile)
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserProfile(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
    )
