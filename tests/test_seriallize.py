"""
Pytest suite for the serializers found in pycore.serialize.

Run with:
    pytest tests/test_serializers.py -v
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from enum import Enum
from uuid import UUID
from typing import List, Optional
from dataclasses import dataclass, field

from pycore.serialize.compressed import CompressionAlgo, CompressedSerializer
from pycore.serialize.dataclass import DataclassSerializer
from pycore.serialize.encrypted import EncryptedSerializer
from pycore.serialize.msgpack import MsgPackSerializer
from pycore.serialize.versioned import VersionedSerializer


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def msgpack_serializer():
    """Provide a fresh MsgPackSerializer instance."""
    return MsgPackSerializer()


@pytest.fixture
def sample_data():
    """Provide a complex dictionary with various types."""
    return {
        "name": "Alice",
        "score": Decimal("98.765"),
        "joined": date(2023, 6, 1),
        "last_login": datetime(2024, 1, 15, 10, 30, 45, 123_000),
        "session_id": UUID("12345678-1234-5678-1234-567812345678"),
        "tags": {"admin", "user"},
        "coords": (48.8566, 2.3522),
        "nested": {"inner_set": {1, 2, 3}},
    }


# ------------------------------------------------------------------------------
# MsgPackSerializer tests
# ------------------------------------------------------------------------------

class TestMsgPackSerializer:
    """Tests for MsgPackSerializer."""

    def test_roundtrip_complex_data(self, msgpack_serializer, sample_data):
        """Complex data survives serialization round‑trip."""
        raw = msgpack_serializer.serialize(sample_data)
        restored = msgpack_serializer.deserialize(raw)

        assert restored["score"] == Decimal("98.765")
        assert restored["joined"] == date(2023, 6, 1)
        assert restored["last_login"] == datetime(2024, 1, 15, 10, 30, 45, 123_000)
        assert restored["session_id"] == UUID("12345678-1234-5678-1234-567812345678")
        assert restored["tags"] == {"admin", "user"}
        assert restored["coords"] == (48.8566, 2.3522)
        assert restored["nested"]["inner_set"] == {1, 2, 3}

    def test_determinism(self, msgpack_serializer, sample_data):
        """Same data produces identical bytes."""
        first = msgpack_serializer.serialize(sample_data)
        second = msgpack_serializer.serialize(sample_data)
        assert first == second

    def test_datetime_date_type_distinction(self, msgpack_serializer):
        """datetime and date types are preserved correctly."""
        data = {"ts": datetime(2024, 3, 1, 12, 0), "d": date(2024, 3, 1)}
        restored = msgpack_serializer.roundtrip(data)
        assert type(restored["ts"]) is datetime
        assert type(restored["d"]) is date

    def test_instance_isolation(self):
        """Each MsgPackSerializer instance has its own ext_encoders dict."""
        s1 = MsgPackSerializer()
        s2 = MsgPackSerializer()
        s1._ext_encoders[int] = lambda v: None  # dummy mutation
        assert int not in s2._ext_encoders


# ------------------------------------------------------------------------------
# VersionedSerializer tests
# ------------------------------------------------------------------------------

class TestVersionedSerializer:
    """Tests for VersionedSerializer with migrations."""

    @pytest.fixture
    def base_serializer(self):
        return MsgPackSerializer()

    @pytest.fixture
    def v3_serializer(self, base_serializer):
        """Serializer at version 3 with v1→v2 and v2→v3 migrations."""
        ser = VersionedSerializer(base=base_serializer, current_version=3)
        ser.register_migration(1, lambda d: {**d, "country": "unknown"})
        ser.register_migration(2, lambda d: {**d, "full_name": d.pop("name", "")})
        return ser

    def test_current_version_roundtrip(self, v3_serializer):
        """Data written at current version round‑trips unchanged."""
        current = {"full_name": "Alice", "age": 30, "country": "US"}
        payload = v3_serializer.serialize(current)
        restored = v3_serializer.deserialize(payload)
        assert restored == current

    def test_migration_v1_to_v3(self, base_serializer, v3_serializer):
        """A v1 payload is correctly migrated to v3."""
        v1_ser = VersionedSerializer(base=base_serializer, current_version=1)
        v1_payload = v1_ser.serialize({"name": "Bob", "age": 25})

        migrated = v3_serializer.deserialize(v1_payload)
        assert migrated["full_name"] == "Bob"
        assert migrated["country"] == "unknown"
        assert migrated["age"] == 25

    def test_migration_v2_to_v3(self, base_serializer, v3_serializer):
        """A v2 payload is correctly migrated to v3."""
        v2_ser = VersionedSerializer(base=base_serializer, current_version=2)
        v2_payload = v2_ser.serialize({"name": "Carol", "age": 28, "country": "FR"})

        migrated = v3_serializer.deserialize(v2_payload)
        assert migrated["full_name"] == "Carol"
        assert migrated["country"] == "FR"
        assert migrated["age"] == 28

    def test_future_version_raises(self, base_serializer, v3_serializer):
        """Deserializing a version newer than current raises ValueError."""
        future_ser = VersionedSerializer(base=base_serializer, current_version=99)
        future_payload = future_ser.serialize({"x": 1})

        with pytest.raises(ValueError, match="99"):
            v3_serializer.deserialize(future_payload)

    def test_missing_migration_raises(self, base_serializer):
        """Deserializing with incomplete migration chain raises ValueError."""
        broken = VersionedSerializer(base=base_serializer, current_version=3)
        broken.register_migration(1, lambda d: d)  # only v1→v2 defined
        v1_payload = VersionedSerializer(base=base_serializer, current_version=1).serialize({"x": 1})

        with pytest.raises(ValueError, match="migration"):
            broken.deserialize(v1_payload)


# ------------------------------------------------------------------------------
# CompressedSerializer tests
# ------------------------------------------------------------------------------

class TestCompressedSerializer:
    """Tests for CompressedSerializer with various algorithms."""

    @pytest.fixture
    def base_serializer(self):
        return MsgPackSerializer()

    @pytest.fixture
    def large_data(self):
        return {"text": "hello world " * 200, "nums": list(range(100))}

    @pytest.mark.parametrize("algo", CompressionAlgo)
    def test_roundtrip_and_compression(self, base_serializer, large_data, algo):
        """Each compression algorithm round‑trips and reduces size."""
        cs = CompressedSerializer(base=base_serializer, algorithm=algo, threshold=0)
        raw = cs.serialize(large_data)
        restored = cs.deserialize(raw)

        assert restored == large_data

        stats = cs.stats(large_data)
        assert stats["savings_pct"] > 0

    def test_below_threshold_stored_uncompressed(self, base_serializer):
        """Payloads smaller than threshold use the no‑compression path."""
        cs = CompressedSerializer(base=base_serializer, threshold=10_000)
        small = {"x": 1}
        raw = cs.serialize(small)

        assert raw[0] == 0x00  # no‑compression magic byte
        assert cs.deserialize(raw) == small

    def test_cross_algorithm_auto_detection(self, base_serializer):
        """Compressed payloads can be decoded by a serializer with a different algo."""
        data = {"key": "value " * 50}
        zlib_cs = CompressedSerializer(base=base_serializer, algorithm=CompressionAlgo.ZLIB, threshold=0)
        gzip_cs = CompressedSerializer(base=base_serializer, algorithm=CompressionAlgo.GZIP, threshold=0)

        zlib_payload = zlib_cs.serialize(data)
        gzip_payload = gzip_cs.serialize(data)

        assert gzip_cs.deserialize(zlib_payload) == data
        assert zlib_cs.deserialize(gzip_payload) == data


# ------------------------------------------------------------------------------
# EncryptedSerializer tests
# ------------------------------------------------------------------------------

class TestEncryptedSerializer:
    """Tests for EncryptedSerializer using Fernet encryption."""

    @pytest.fixture
    def base_serializer(self):
        return MsgPackSerializer()

    @pytest.fixture
    def secret_data(self):
        return {"secret": "top-secret-value", "amount": Decimal("9999.99")}

    @pytest.fixture
    def encryption_key(self):
        return EncryptedSerializer.generate_key()

    def test_raw_key_roundtrip(self, base_serializer, secret_data, encryption_key):
        """Encryption with a raw key round‑trips correctly."""
        enc = EncryptedSerializer(base=base_serializer, key=encryption_key)
        payload = enc.serialize(secret_data)
        restored = enc.deserialize(payload)

        assert restored["secret"] == "top-secret-value"
        assert restored["amount"] == Decimal("9999.99")

    def test_wrong_key_raises(self, base_serializer, secret_data, encryption_key):
        """Decryption with the wrong key raises ValueError."""
        enc = EncryptedSerializer(base=base_serializer, key=encryption_key)
        payload = enc.serialize(secret_data)

        wrong_key = EncryptedSerializer.generate_key()
        wrong_enc = EncryptedSerializer(base=base_serializer, key=wrong_key)

        with pytest.raises(ValueError, match="wrong key|tampered"):
            wrong_enc.deserialize(payload)

    def test_tampered_ciphertext_raises(self, base_serializer, secret_data, encryption_key):
        """Tampering with the ciphertext causes a ValueError."""
        enc = EncryptedSerializer(base=base_serializer, key=encryption_key)
        payload = enc.serialize(secret_data)

        tampered = bytearray(payload)
        tampered[len(payload) // 2] ^= 0xFF

        with pytest.raises(ValueError):
            enc.deserialize(bytes(tampered))

    def test_password_derived_key_roundtrip(self, base_serializer, secret_data):
        """Encryption using a password‑derived key works."""
        password = "correct-horse-battery-staple"
        enc = EncryptedSerializer.from_password(base=base_serializer, password=password)
        payload = enc.serialize(secret_data)
        restored = enc.deserialize(payload)

        assert restored == secret_data

    def test_one_shot_decrypt_with_password(self, base_serializer, secret_data):
        """One‑shot decryption works with the correct password."""
        password = "correct-horse-battery-staple"
        enc = EncryptedSerializer.from_password(base=base_serializer, password=password)
        payload = enc.serialize(secret_data)

        decrypted = EncryptedSerializer.decrypt_with_password(
            base_serializer, payload, password
        )
        assert decrypted == secret_data

    def test_ciphertext_randomized(self, base_serializer, secret_data):
        """Each encryption produces a different ciphertext (semantic security)."""
        enc = EncryptedSerializer.from_password(base_serializer, password="pass")
        p1 = enc.serialize(secret_data)
        p2 = enc.serialize(secret_data)

        assert p1 != p2
        assert enc.deserialize(p1) == enc.deserialize(p2)


# ------------------------------------------------------------------------------
# DataclassSerializer tests
# ------------------------------------------------------------------------------

class TestDataclassSerializer:
    """Tests for DataclassSerializer with complex nested dataclasses."""

    @pytest.fixture
    def dataclass_serializer(self):
        return DataclassSerializer()

    @pytest.fixture
    def sample_user(self):
        """Create a User instance for testing."""
        class Role(Enum):
            ADMIN = "admin"
            USER = "user"

        @dataclass
        class Address:
            street: str
            city: str
            zip_code: str

        @dataclass
        class User:
            id: UUID
            name: str
            role: Role
            joined: date
            last_login: Optional[datetime]
            scores: List[Decimal]
            address: Address
            tags: List[str] = field(default_factory=list)

        # Make classes accessible for later use
        self.Role = Role
        self.Address = Address
        self.User = User

        return User(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            name="Alice",
            role=Role.ADMIN,
            joined=date(2023, 6, 1),
            last_login=datetime(2024, 1, 15, 10, 30),
            scores=[Decimal("9.5"), Decimal("8.75")],
            address=Address(street="123 Main St", city="Springfield", zip_code="12345"),
            tags=["python", "backend"],
        )

    def test_full_user_roundtrip(self, dataclass_serializer, sample_user):
        """Complex dataclass with many types survives round‑trip."""
        raw = dataclass_serializer.serialize(sample_user)
        restored = dataclass_serializer.deserialize(raw, target_type=self.User)

        assert restored.id == sample_user.id
        assert restored.role == self.Role.ADMIN
        assert restored.joined == date(2023, 6, 1)
        assert restored.last_login == datetime(2024, 1, 15, 10, 30)
        assert restored.scores == [Decimal("9.5"), Decimal("8.75")]
        assert restored.address.city == "Springfield"
        assert restored.tags == ["python", "backend"]

    def test_optional_none_handling(self, dataclass_serializer):
        """Optional fields with value None are correctly serialized."""
        @dataclass
        class Simple:
            id: UUID
            last_login: Optional[datetime]

        user = Simple(
            id=UUID("00000000-0000-0000-0000-000000000000"),
            last_login=None,
        )

        restored = dataclass_serializer.roundtrip_typed(user, Simple)
        assert restored.last_login is None

    def test_strict_mode_rejects_extra_fields(self, dataclass_serializer, sample_user):
        """Strict mode raises ValueError when unknown fields are present."""
        strict = DataclassSerializer(strict=True)
        raw = strict.serialize(sample_user)

        # Inject an extra field into the msgpack payload
        inner = strict._base.deserialize(raw)
        inner["__extra__"] = "injected"
        tampered_raw = strict._base.serialize(inner)

        with pytest.raises(ValueError, match="__extra__"):
            strict.deserialize(tampered_raw, target_type=self.User)


# ------------------------------------------------------------------------------
# Composition test
# ------------------------------------------------------------------------------

def test_composition_pipeline():
    """Compressed(Encrypted(MsgPack)) pipeline works end‑to‑end."""
    key = EncryptedSerializer.generate_key()
    pipeline = CompressedSerializer(
        base=EncryptedSerializer(
            base=MsgPackSerializer(),
            key=key,
        ),
        algorithm=CompressionAlgo.ZLIB,
        threshold=0,
    )

    data = {
        "users": [
            {"id": i, "name": f"user_{i}", "score": Decimal(str(i * 1.5))}
            for i in range(50)
        ],
        "timestamp": datetime(2024, 6, 1, 12, 0),
        "session": UUID("aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"),
    }

    raw = pipeline.serialize(data)
    restored = pipeline.deserialize(raw)

    assert restored["timestamp"] == datetime(2024, 6, 1, 12, 0)
    assert restored["session"] == UUID("aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb")
    assert len(restored["users"]) == 50
    assert restored["users"][5]["score"] == Decimal("7.5")


# ------------------------------------------------------------------------------
# Helper monkey-patch (required for roundtrip_typed test above)
# ------------------------------------------------------------------------------

def _roundtrip_typed(self, data, target_type):
    return self.deserialize(self.serialize(data), target_type=target_type)

# Apply the helper if it hasn't been already
if not hasattr(DataclassSerializer, "roundtrip_typed"):
    DataclassSerializer.roundtrip_typed = _roundtrip_typed