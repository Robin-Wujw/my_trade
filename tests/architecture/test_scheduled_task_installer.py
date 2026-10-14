"""Exercise Windows task actions without registering or running a real task."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
POWERSHELL = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not POWERSHELL,
    reason="Windows PowerShell is required",
)


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout with spaces and 'quote'"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("setup_scheduled_task.bat", "setup_scheduled_task.ps1"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    (scripts / "run_daily_analysis.ps1").write_text(
        "Set-Content -LiteralPath $env:ACTION_RESULT -Value $PSScriptRoot\nexit 17\n",
        encoding="utf-8",
    )
    return root


def run_installer(checkout, *, fail_registration=False):
    state_path = checkout / "registered-task.json"
    state_path.write_text('{"previous_task": true}', encoding="utf-8")
    harness = checkout / "mock_task_registration.ps1"
    harness.write_text(
        r"""
$ErrorActionPreference = 'Stop'
# Mock the CIM-backed cmdlets as well: some Windows test accounts cannot access
# that provider. The captured action is still launched in a real native process.
function New-ScheduledTaskAction {
    param($Execute, $Argument, $WorkingDirectory)
    [pscustomobject]@{Execute = $Execute; Arguments = $Argument; WorkingDirectory = $WorkingDirectory}
}
function New-ScheduledTaskTrigger {
    param([switch]$Daily, [datetime]$At)
    if (-not $Daily) { throw 'Expected a daily schedule' }
    [pscustomobject]@{StartBoundary = $At.ToString('s'); DaysInterval = 1}
}
function New-ScheduledTaskSettingsSet {
    param([switch]$AllowStartIfOnBatteries, [switch]$DontStopIfGoingOnBatteries,
          [switch]$StartWhenAvailable, [switch]$RunOnlyIfNetworkAvailable, [timespan]$ExecutionTimeLimit)
    [pscustomobject]@{ExecutionTimeLimit = [System.Xml.XmlConvert]::ToString($ExecutionTimeLimit)}
}
function Register-ScheduledTask {
    [CmdletBinding()]
    param($TaskName, $Action, $Trigger, $Settings, $Description, $User, $RunLevel, [switch]$Force)
    if ($env:FAIL_REGISTRATION -eq '1') {
        Write-Error 'simulated-registration-denied'
    }
    @{
        task_name = $TaskName
        execute = $Action.Execute
        arguments = $Action.Arguments
        working_directory = $Action.WorkingDirectory
        trigger_start = $Trigger.StartBoundary
        trigger_interval = $Trigger.DaysInterval
        execution_time_limit = $Settings.ExecutionTimeLimit
        user = $User
        run_level = $RunLevel
        force = $Force.IsPresent
    } | ConvertTo-Json | Set-Content -LiteralPath $env:TASK_STATE -Encoding UTF8
}
function Unregister-ScheduledTask {
    [CmdletBinding(SupportsShouldProcess)]
    param($TaskName)
    Set-Content -LiteralPath $env:DELETE_MARKER -Value $TaskName
    throw 'Deleting a task before replacement is forbidden'
}
& $env:SETUP_SCRIPT
exit $LASTEXITCODE
""",
        encoding="utf-8",
    )
    env = os.environ | {
        "SETUP_SCRIPT": str(checkout / "scripts" / "setup_scheduled_task.ps1"),
        "TASK_STATE": str(state_path),
        "DELETE_MARKER": str(checkout / "deleted-task.txt"),
        "FAIL_REGISTRATION": "1" if fail_registration else "0",
    }
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert not (checkout / "deleted-task.txt").exists()
    return result, json.loads(state_path.read_text(encoding="utf-8-sig"))


def test_installer_registers_an_action_that_launches_paths_with_spaces(checkout):
    result, task = run_installer(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert task["task_name"] == "StockSelection-Daily"
    assert task["working_directory"] == str(checkout)
    assert task["trigger_start"][11:16] == "20:30"
    assert task["trigger_interval"] == 1
    assert task["execution_time_limit"] == "PT12H"
    assert task["force"] is True
    assert task["run_level"] == "Highest"
    assert task["user"]

    action_result = checkout / "action-result.txt"
    # Pass the persisted native command line exactly as Task Scheduler would.
    # The target is an isolated fixture, never the production pipeline.
    action = subprocess.run(
        f'"{task["execute"]}" {task["arguments"]}',
        cwd=task["working_directory"],
        env=os.environ | {"ACTION_RESULT": str(action_result)},
        capture_output=True,
        timeout=30,
    )
    assert action.returncode == 17, action.stdout + action.stderr
    assert action_result.read_text().strip() == str(checkout / "scripts")


def test_registration_failure_preserves_previous_task_and_writes_diagnostics(checkout):
    result, task = run_installer(checkout, fail_registration=True)
    assert result.returncode == 1
    assert task == {"previous_task": True}
    assert b"simulated-registration-denied" in result.stderr
    assert b"Task created or updated" not in result.stdout
    logs = list((checkout / "var" / "logs").glob("scheduled_task_setup_*.log"))
    assert len(logs) == 1
    assert "simulated-registration-denied" in logs[0].read_text(encoding="utf-8-sig")


def test_missing_pipeline_script_fails_before_registering(checkout):
    (checkout / "scripts" / "run_daily_analysis.ps1").unlink()
    result, task = run_installer(checkout)
    assert result.returncode == 1
    assert task == {"previous_task": True}
    assert b"Daily pipeline script does not exist" in result.stderr


@pytest.mark.parametrize("exit_code", [0, 23])
def test_batch_launcher_preserves_installer_exit_code(checkout, exit_code):
    # Replace only the isolated fixture's installer; no system task is registered.
    (checkout / "scripts" / "setup_scheduled_task.ps1").write_text(
        "param([int]$RequestedExitCode)\nexit $RequestedExitCode\n", encoding="utf-8"
    )
    launcher = checkout / "scripts" / "setup_scheduled_task.bat"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-RequestedExitCode", str(exit_code)],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
