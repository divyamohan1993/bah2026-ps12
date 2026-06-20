"""Content-addressed cache for on-demand interpolation (SERVE, research/06 §3.3, §6.3).

The on-demand ``/interpolate`` endpoint is the optional fallback to the precomputed O(1)
path. Because interpolation is deterministic in ``(I0, I1, t, model_version)``, every result
can be cached under a content hash of exactly those inputs:

    key = sha256(I0_bytes || I1_bytes || t || model_version)

A keyed ``GET`` is then O(1). The store is **disk-backed by default** (one file per key
under a cache dir) with an **optional Redis** front (lazily connected; if Redis is
unavailable or errors, it transparently falls back to disk). This mirrors the "Redis caches
small payloads; CDN/disk holds the bytes" tiering in research/03 §6.3.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


def _array_digest(arr: np.ndarray) -> bytes:
    """Return a deterministic byte signature for an array (shape + dtype + C-order bytes)."""
    import numpy as np

    a = np.ascontiguousarray(np.asarray(arr))
    header = f"{a.dtype.str}|{a.shape}".encode()
    return header + a.tobytes(order="C")


def key(
    i0: np.ndarray,
    i1: np.ndarray,
    t: float,
    model_ver: str,
) -> str:
    """Compute the content-addressed cache key for an interpolation request.

    Args:
        i0: earlier bracket frame (array).
        i1: later bracket frame (array).
        t: interpolation fraction in (0, 1).
        model_ver: model/checkpoint version string (so a new model invalidates old entries).

    Returns:
        A hex ``sha256`` digest string uniquely identifying ``(i0, i1, t, model_ver)``.
    """
    h = hashlib.sha256()
    h.update(b"frameflow-interp-v1\x00")
    h.update(_array_digest(i0))
    h.update(b"\x00")
    h.update(_array_digest(i1))
    h.update(b"\x00")
    # Format t with fixed precision so 0.5 and 0.5000001 don't collide unexpectedly while
    # bit-identical floats hash identically.
    h.update(f"t={float(t):.9g}".encode())
    h.update(b"\x00")
    h.update(f"model={model_ver}".encode())
    return h.hexdigest()


class InterpolationCache:
    """A content-addressed cache: disk-backed, with an optional lazy Redis front.

    Usage::

        cache = InterpolationCache(cache_dir="/tmp/ff-cache", redis_url=None)
        k = cache.make_key(i0, i1, t=0.5, model_ver="v0.1.0")
        if (hit := cache.get(k)) is None:
            hit = run_model(...)            # bytes (e.g. a .nc or PNG)
            cache.set(k, hit)

    Redis is connected lazily on first use; any Redis error degrades silently to disk-only
    so the endpoint never fails because the cache backend is down.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/frameflow/interp",
        redis_url: str | None = None,
        namespace: str = "frameflow:interp",
    ) -> None:
        """Initialize the cache.

        Args:
            cache_dir: directory for the disk-backed store (created on demand).
            redis_url: optional Redis URL (e.g. ``redis://localhost:6379/0``). If ``None``
                (or connection fails) the cache is disk-only.
            namespace: key prefix used for Redis entries.
        """
        self.cache_dir = Path(cache_dir)
        self.redis_url = redis_url
        self.namespace = namespace
        self._redis: Any = None
        self._redis_tried = False

    # -- key helpers -----------------------------------------------------------------------
    @staticmethod
    def make_key(i0: np.ndarray, i1: np.ndarray, t: float, model_ver: str) -> str:
        """Content-addressed key for ``(i0, i1, t, model_ver)`` (see module-level :func:`key`)."""
        return key(i0, i1, t, model_ver)

    def _path_for(self, k: str) -> Path:
        return self.cache_dir / f"{k}.bin"

    def _redis_key(self, k: str) -> str:
        return f"{self.namespace}:{k}"

    # -- optional redis --------------------------------------------------------------------
    def _get_redis(self) -> Any:
        """Lazily connect to Redis; cache the (possibly ``None``) client. Never raises."""
        if self._redis_tried:
            return self._redis
        self._redis_tried = True
        if not self.redis_url:
            self._redis = None
            return None
        try:  # pragma: no cover - exercised only when a Redis server is reachable
            import redis  # lazy

            client = redis.Redis.from_url(self.redis_url, socket_connect_timeout=0.5)
            client.ping()
            self._redis = client
        except Exception:
            self._redis = None
        return self._redis

    # -- public API ------------------------------------------------------------------------
    def get(self, k: str) -> bytes | None:
        """Return cached bytes for key ``k`` (Redis first, then disk), or ``None`` if absent."""
        client = self._get_redis()
        if client is not None:
            try:  # pragma: no cover - needs live Redis
                val = client.get(self._redis_key(k))
                if val is not None:
                    return bytes(val)
            except Exception:
                pass  # fall through to disk
        p = self._path_for(k)
        if p.exists():
            return p.read_bytes()
        return None

    def set(self, k: str, value: bytes, ttl_seconds: int | None = None) -> None:
        """Store ``value`` under key ``k`` on disk (and in Redis if available).

        Args:
            k: content-addressed key.
            value: payload bytes (e.g. an encoded ``.nc`` or PNG).
            ttl_seconds: optional Redis TTL; disk entries are persistent.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        p = self._path_for(k)
        # Atomic-ish write: write to a temp sibling then replace.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(value)
        os.replace(tmp, p)

        client = self._get_redis()
        if client is not None:
            try:  # pragma: no cover - needs live Redis
                if ttl_seconds:
                    client.set(self._redis_key(k), value, ex=int(ttl_seconds))
                else:
                    client.set(self._redis_key(k), value)
            except Exception:
                pass  # disk write already succeeded

    def has(self, k: str) -> bool:
        """Return True if key ``k`` is present in Redis or on disk."""
        client = self._get_redis()
        if client is not None:
            try:  # pragma: no cover - needs live Redis
                if client.exists(self._redis_key(k)):
                    return True
            except Exception:
                pass
        return self._path_for(k).exists()

    def clear(self) -> None:
        """Remove all disk entries (Redis namespace is left untouched)."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.bin"):
                try:
                    f.unlink()
                except OSError:  # pragma: no cover - best effort
                    pass


__all__ = ["key", "InterpolationCache"]
