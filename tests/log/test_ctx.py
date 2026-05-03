import logging
import pytest
from unittest.mock import MagicMock, patch

from pycore.log.ctx import ContextAdapter


@pytest.fixture
def mock_logger():
    """A MagicMock that stands in for a real Logger."""
    logger = MagicMock(spec=logging.Logger)
    logger.manager = MagicMock()
    logger.manager.disable = 0
    logger.isEnabledFor = MagicMock(return_value=True)
    return logger


@pytest.fixture
def adapter(mock_logger):
    return ContextAdapter(mock_logger, {})


# ── process() unit tests ──────────────────────────────────────────────────────

class TestProcess:
    def test_no_context_returns_message_unchanged(self, adapter):
        msg, kwargs = adapter.process("hello", {})
        assert msg == "hello"
        assert kwargs == {}

    def test_extra_dict_appended_to_message(self, adapter):
        msg, kwargs = adapter.process("login", {"extra": {"user_id": 42}})
        assert msg == "login | user_id=42"
        assert kwargs == {}

    def test_direct_kwargs_appended_to_message(self, adapter):
        msg, kwargs = adapter.process("login", {"user_id": 42})
        assert msg == "login | user_id=42"
        assert kwargs == {}

    def test_extra_and_direct_kwargs_are_merged(self, adapter):
        msg, kwargs = adapter.process(
            "event", {"extra": {"source": "web"}, "user_id": 7}
        )
        # Both keys must appear; order may vary so check parts individually
        assert msg.startswith("event | ")
        assert "source=web" in msg
        assert "user_id=7" in msg
        assert kwargs == {}

    def test_extra_key_is_removed_from_returned_kwargs(self, adapter):
        incoming = {"extra": {"k": "v"}}
        _, returned_kwargs = adapter.process("msg", incoming)
        assert "extra" not in returned_kwargs

    def test_multiple_context_keys_all_present(self, adapter):
        ctx = {"extra": {"a": 1, "b": 2, "c": 3}}
        msg, _ = adapter.process("m", ctx)
        for pair in ("a=1", "b=2", "c=3"):
            assert pair in msg

    def test_empty_extra_dict_no_context_suffix(self, adapter):
        msg, kwargs = adapter.process("bare", {"extra": {}})
        assert msg == "bare"

    def test_non_string_values_formatted_correctly(self, adapter):
        msg, _ = adapter.process("data", {"extra": {"count": 0, "ratio": 3.14}})
        assert "count=0" in msg
        assert "ratio=3.14" in msg

    def test_message_format_separator(self, adapter):
        msg, _ = adapter.process("base", {"extra": {"x": 1}})
        assert " | " in msg

    def test_empty_kwargs_passthrough(self, adapter):
        """When no context at all, original kwargs dict is returned as-is."""
        original = {}
        msg, returned = adapter.process("no-ctx", original)
        assert msg == "no-ctx"
        assert returned is original


# ── integration: real logging call goes through process() ────────────────────

class TestLoggingIntegration:
    # In Python 3.12 LoggerAdapter routes every call through
    # self.logger.log(level, msg, ...) — not the level-specific methods.

    def test_info_call_formats_message(self, mock_logger):
        """adapter.info() should ultimately call logger.log with formatted msg."""
        adapter = ContextAdapter(mock_logger, {})
        adapter.info("User logged in", extra={"user_id": 123})
        mock_logger.log.assert_called_once()
        _level, msg = mock_logger.log.call_args[0][:2]
        assert "user_id=123" in msg

    def test_warning_call_formats_message(self, mock_logger):
        adapter = ContextAdapter(mock_logger, {})
        adapter.warning("Low disk", extra={"free_gb": 2})
        mock_logger.log.assert_called_once()
        _level, msg = mock_logger.log.call_args[0][:2]
        assert "free_gb=2" in msg

    def test_error_call_formats_message(self, mock_logger):
        adapter = ContextAdapter(mock_logger, {})
        adapter.error("Crash", extra={"code": 500})
        mock_logger.log.assert_called_once()
        _level, msg = mock_logger.log.call_args[0][:2]
        assert "code=500" in msg

    def test_no_extra_message_unchanged(self, mock_logger):
        adapter = ContextAdapter(mock_logger, {})
        adapter.info("plain message")
        mock_logger.log.assert_called_once()
        _level, msg = mock_logger.log.call_args[0][:2]
        assert msg == "plain message"

    def test_correct_level_forwarded_for_warning(self, mock_logger):
        adapter = ContextAdapter(mock_logger, {})
        adapter.warning("w", extra={"k": "v"})
        level = mock_logger.log.call_args[0][0]
        assert level == logging.WARNING

    def test_correct_level_forwarded_for_error(self, mock_logger):
        adapter = ContextAdapter(mock_logger, {})
        adapter.error("e", extra={"k": "v"})
        level = mock_logger.log.call_args[0][0]
        assert level == logging.ERROR

    def test_real_logger_captures_output(self, caplog):
        """End-to-end: verify output reaches the Python logging system."""
        real_logger = logging.getLogger("test.context_adapter")
        adapter = ContextAdapter(real_logger, {})
        with caplog.at_level(logging.INFO, logger="test.context_adapter"):
            adapter.info("login", extra={"user": "alice"})
        assert any("user=alice" in r.message for r in caplog.records)


# ── edge-case / boundary tests ────────────────────────────────────────────────

class TestEdgeCases:
    def test_none_value_in_context(self, adapter):
        msg, _ = adapter.process("m", {"extra": {"key": None}})
        assert "key=None" in msg

    def test_boolean_value_in_context(self, adapter):
        msg, _ = adapter.process("m", {"extra": {"flag": True}})
        assert "flag=True" in msg

    def test_empty_string_message(self, adapter):
        msg, _ = adapter.process("", {"extra": {"k": "v"}})
        assert msg == " | k=v"

    def test_message_with_special_characters(self, adapter):
        msg, _ = adapter.process("msg: <ok>", {"extra": {"x": "y"}})
        assert msg.startswith("msg: <ok> | ")

    def test_direct_kwargs_no_extra_key(self, adapter):
        """Kwargs without 'extra' are treated as context directly."""
        msg, kwargs = adapter.process("ev", {"status": "ok"})
        assert "status=ok" in msg
        assert kwargs == {}