"""
EncryptedSerializer

Wraps any serializer and applies authenticated encryption.

Algorithm: Fernet = AES-128-CBC + HMAC-SHA256, per cryptography.load.
  - Provides both confidentiality and integrity (tamper-evident).
  - A tampered ciphertext raises InvalidToken rather than silently producing
    garbage — critical for security-sensitive deserialization (OWASP guideline).

Key handling:
  - Accept a raw 32-byte Fernet key directly, OR
  - Derive one from a human password via PBKDF2-HMAC-SHA256 with a random salt.
    The salt is stored alongside the ciphertext so the key can be re-derived
    on deserialization without any out-of-band key exchange.

Wire format (when using password-derived key):
  [4-byte salt-length][salt bytes][fernet ciphertext]

Wire format (when using raw key):
  [fernet ciphertext]  (no salt prefix — key is managed externally)
"""
import base64
import os
import struct
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from pycore.serialize.base import BaseSerializer


class EncryptedSerializer(BaseSerializer):
    """
    Authenticated encryption wrapper around any BaseSerializer.

    Uses Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` package.
    A tampered or corrupted ciphertext raises `InvalidToken` immediately —
    no silent data corruption.

    Usage (password-derived key — easiest for most use cases)
    ----------------------------------------------------------
        ser = EncryptedSerializer.from_password(
            base=MsgPackSerializer(),
            password="correct-horse-battery-staple",
        )
        payload = ser.serialize({"secret": 42})
        data    = ser.deserialize(payload)

    Usage (raw Fernet key — preferred for production key rotation)
    -------------------------------------------------------------
        key = EncryptedSerializer.generate_key()
        ser = EncryptedSerializer(base=MsgPackSerializer(), key=key)
    """

    _SALT_LENGTH = 16  # bytes
    _KDF_ITERATIONS = 480_000  # OWASP 2023 recommendation for PBKDF2-SHA256

    def __init__(
            self,
            base: BaseSerializer,
            key: bytes,
            *,
            _include_salt: bool = False,
            _salt: bytes | None = None,
    ) -> None:
        self._base = base
        self._fernet = Fernet(key)
        # Salt fields used only when key was password-derived
        self._include_salt = _include_salt
        self._salt = _salt

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new random 32-byte URL-safe base64-encoded Fernet key."""
        return Fernet.generate_key()

    @classmethod
    def from_password(
            cls,
            base: BaseSerializer,
            password: str,
            salt: bytes | None = None,
    ) -> "EncryptedSerializer":
        """
        Derive a Fernet key from a human-readable password via PBKDF2-SHA256.
        If `salt` is None, a new random salt is generated (for new keys).
        Pass the same salt to re-derive the same key (for decryption).
        """
        salt = salt or os.urandom(cls._SALT_LENGTH)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=cls._KDF_ITERATIONS,
        )
        raw_key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
        return cls(base, raw_key, _include_salt=True, _salt=salt)

    def serialize(self, data: Any) -> bytes:
        inner = self._base.serialize(data)
        ciphertext = self._fernet.encrypt(inner)
        if self._include_salt and self._salt:
            # Prepend 4-byte length-prefixed salt so deserialize can re-derive key
            salt_len = struct.pack(">I", len(self._salt))
            return salt_len + self._salt + ciphertext
        return ciphertext

    def deserialize(self, raw: bytes) -> Any:
        try:
            if self._include_salt and self._salt:
                # Strip the salt prefix we wrote during serialize
                (salt_len,) = struct.unpack(">I", raw[:4])
                ciphertext = raw[4 + salt_len:]
            else:
                ciphertext = raw
            plaintext = self._fernet.decrypt(ciphertext)
        except InvalidToken as exc:
            raise ValueError(
                "EncryptedSerializer: decryption failed — "
                "wrong key or tampered ciphertext"
            ) from exc
        return self._base.deserialize(plaintext)

    @classmethod
    def decrypt_with_password(
            cls,
            base: BaseSerializer,
            raw: bytes,
            password: str,
    ) -> Any:
        """
        One-shot decryption when you have a payload but no pre-built instance.
        Reads the salt from the payload prefix and re-derives the key.
        """
        (salt_len,) = struct.unpack(">I", raw[:4])
        salt = raw[4: 4 + salt_len]
        instance = cls.from_password(base, password, salt=salt)
        return instance.deserialize(raw)
