import importlib
from unittest.mock import MagicMock, patch

import pytest

from pycore import cpu


@patch('psutil.Process')
def test_macos_fallback(mock_process_cls):
    """
    Scenario: Running on M1 Mac.
    Expectation: Do NOT call cpu_affinity. DO call nice().
    """
    mock_proc_instance = MagicMock()
    del mock_proc_instance.cpu_affinity
    mock_process_cls.return_value = mock_proc_instance

    with patch.object(cpu_utils, 'IS_MACOS', False), \
         patch.object(cpu_utils, 'IS_WINDOWS', False):
        result = cpu_utils.set_high_priority("TestWorker")

    assert result is True
    mock_proc_instance.nice.assert_called_with(-10)
    assert not hasattr(mock_proc_instance, 'cpu_affinity') \
        or not mock_proc_instance.cpu_affinity.called


@patch('psutil.Process')
def test_linux_pinning(mock_process_cls):
    """
    Scenario: Running on Linux (simulated).
    Expectation: CALL cpu_affinity.
    """
    mock_proc_instance = MagicMock()
    mock_process_cls.return_value = mock_proc_instance

    with patch.object(cpu_utils, 'IS_MACOS', False), \
         patch.object(cpu_utils, 'IS_WINDOWS', False), \
         patch('psutil.cpu_count', return_value=8):
        result = cpu_utils.set_cpu_affinity([0, 1], "TestWorker")

    assert result is True
    mock_proc_instance.cpu_affinity.assert_called_with([0, 1])


@patch('psutil.Process')
def test_invalid_core_protection(mock_process_cls):
    """
    Scenario: Asking for Core 99 on a 4-core machine.
    Expectation: Return False, do not crash.
    """
    mock_proc_instance = MagicMock()
    mock_process_cls.return_value = mock_proc_instance

    with patch.object(cpu_utils, 'IS_MACOS', False), \
         patch.object(cpu_utils, 'IS_WINDOWS', False), \
         patch('psutil.cpu_count', return_value=4):
        result = cpu_utils.set_cpu_affinity([10], "TestWorker")

    assert result is False
    mock_proc_instance.cpu_affinity.assert_not_called()