from unittest.mock import Mock, call, patch

import pytest
from pymmcore_plus import DeviceType

from cns_control.utils import stop_live_acquisitions, unload


def test_unload_stops_all_cameras_before_disconnect_and_unload():
    core = Mock()
    core.getLoadedDevicesOfType.return_value = ["Camera1", "Camera2"]
    with patch("cns_control.utils.time.sleep"):
        unload(core, N=1)
    core.getLoadedDevicesOfType.assert_called_once_with(DeviceType.CameraDevice)
    calls = core.mock_calls
    assert calls.index(call.stopSequenceAcquisition("Camera1")) < calls.index(
        call.events.channelGroupChanged.disconnect()
    )
    assert calls.index(call.stopSequenceAcquisition("Camera2")) < calls.index(
        call.unloadAllDevices()
    )


def test_stop_failure_attempts_other_cameras_and_prevents_unload():
    core = Mock()
    core.getLoadedDevicesOfType.return_value = ["Camera1", "Camera2"]
    core.stopSequenceAcquisition.side_effect = [RuntimeError("busy"), None]
    with pytest.raises(RuntimeError, match="Camera1: busy"):
        unload(core, N=1)
    assert core.stopSequenceAcquisition.call_args_list == [
        call("Camera1"), call("Camera2")
    ]
    core.unloadAllDevices.assert_not_called()
    core.events.channelGroupChanged.disconnect.assert_not_called()


def test_no_loaded_cameras_needs_no_stop():
    core = Mock()
    core.getLoadedDevicesOfType.return_value = []
    stop_live_acquisitions(core)
    core.stopSequenceAcquisition.assert_not_called()
