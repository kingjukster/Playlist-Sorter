[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Library,

    [string]$Output = "./runs/smoke",

    [ValidateNotNullOrEmpty()]
    [string]$Distribution = "Ubuntu-24.04"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$SupportedExtensions = @(".mp3", ".ogg", ".flac", ".wav", ".m4a")

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)
    $converted = & wsl.exe --distribution $Distribution --exec wslpath -a $WindowsPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "wslpath failed for '$WindowsPath': $converted"
    }
    return ($converted | Out-String).Trim()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string[]]$Lines)
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes(($Lines -join "`n"))
    $Digest = [System.Security.Cryptography.SHA256]::HashData($Bytes)
    return ([System.Convert]::ToHexString($Digest)).ToLowerInvariant()
}

function Get-FileAggregate {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo[]]$Files,
        [Parameter(Mandatory = $true)][string]$Root
    )
    $Records = @(
        $Files | Sort-Object FullName | ForEach-Object {
            $Relative = [System.IO.Path]::GetRelativePath($Root, $_.FullName).Replace("\", "/")
            $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$Relative $Hash"
        }
    )
    return Get-TextSha256 -Lines $Records
}

function Get-RepositoryAggregate {
    param([Parameter(Mandatory = $true)][string]$Repository)
    $RelativeFiles = @(& git -C $Repository ls-files --cached --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) { throw "Cannot enumerate repository files for evidence binding." }
    $Files = @($RelativeFiles | ForEach-Object { Get-Item -LiteralPath (Join-Path $Repository $_) })
    return Get-FileAggregate -Files $Files -Root $Repository
}

$LibraryPath = [System.IO.Path]::GetFullPath($Library)
if (-not (Test-Path -LiteralPath $LibraryPath -PathType Container)) {
    throw "Library must be an explicitly supplied existing directory: $LibraryPath"
}

$OutputPath = [System.IO.Path]::GetFullPath($Output)
$LibraryPrefix = $LibraryPath.TrimEnd([char[]]@("\", "/")) +
    [System.IO.Path]::DirectorySeparatorChar
if ($OutputPath.Equals($LibraryPath, [System.StringComparison]::OrdinalIgnoreCase) -or
    $OutputPath.StartsWith($LibraryPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output must not be the library or a child of it."
}
if ((Test-Path -LiteralPath $OutputPath) -and
    @(Get-ChildItem -LiteralPath $OutputPath -Force).Count -gt 0) {
    throw "Output must be absent or empty so prior evidence is never overwritten: $OutputPath"
}

$AudioFiles = @(
    Get-ChildItem -LiteralPath $LibraryPath -Recurse -File |
        Where-Object { $SupportedExtensions -contains $_.Extension.ToLowerInvariant() }
)
if ($AudioFiles.Count -ne 20) {
    throw "The smoke gate requires exactly 20 supported audio files; found $($AudioFiles.Count)."
}
$InputAggregateSha256 = Get-FileAggregate -Files $AudioFiles -Root $LibraryPath

$RepositoryPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$RepositoryAggregateSha256 = Get-RepositoryAggregate -Repository $RepositoryPath
$WslRepository = Convert-ToWslPath -WindowsPath $RepositoryPath
$WslLibrary = Convert-ToWslPath -WindowsPath $LibraryPath
$WslKernel = (& wsl.exe --distribution $Distribution --exec uname -r 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $WslKernel -notmatch "(?i)WSL2") {
    throw "Distribution '$Distribution' is not verified as WSL2 (kernel: $WslKernel)."
}
$WslHome = (& wsl.exe --distribution $Distribution --exec printenv HOME 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($WslHome)) {
    throw "Cannot resolve HOME in WSL distribution '$Distribution'."
}

$PythonCode = @'
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

source = Path(os.environ["PLAYLIST_SORTER_SOURCE"]) / "playlist_sorter/catalog/scan.py"
spec = importlib.util.spec_from_file_location("playlist_sorter_catalog_scan", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load catalog service from {source}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
scan_library = module.scan_library

root = Path(sys.argv[1])
expected = int(sys.argv[2])
started = time.perf_counter()
songs, failures = scan_library(root)
elapsed = time.perf_counter() - started
meminfo = {}
try:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) / 1024
except (OSError, ValueError, IndexError):
    pass
gpu = None
try:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gpu = query.stdout.strip() or None
except (OSError, subprocess.SubprocessError):
    pass
payload = {
    "available_items": len(songs),
    "expected_items": expected,
    "failure_count": len(failures),
    "failure_codes": sorted(failure.code for failure in failures),
    "gate_passed": len(songs) == expected and not failures,
    "gpu_snapshot": gpu,
    "host_available_ram_mib": meminfo.get("MemAvailable"),
    "items_per_second": len(songs) / elapsed if elapsed else None,
    "peak_process_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "peak_vram_mib": None,
    "peak_vram_reason": "catalog-only CPU gate; no model or CUDA workload",
    "platform": platform.platform(),
    "python": sys.version,
    "seconds": elapsed,
}
print(json.dumps(payload, sort_keys=True))
'@

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
$WslArguments = @(
    "--distribution", $Distribution,
    "--cd", $WslRepository,
    "--exec", "env",
    "PLAYLIST_SORTER_SOURCE=$WslRepository/src",
    "XDG_CACHE_HOME=$WslHome/.cache/playlist-sorter",
    "python3.12", "-c", $PythonCode, $WslLibrary, "20"
)
([ordered]@{ executable = "wsl.exe"; arguments = $WslArguments } |
    ConvertTo-Json -Depth 8) |
    Set-Content -LiteralPath (Join-Path $OutputPath "command.json") -Encoding utf8

$Started = [DateTimeOffset]::UtcNow
$Timer = [System.Diagnostics.Stopwatch]::StartNew()
$RawStdout = & wsl.exe @WslArguments 2> (Join-Path $OutputPath "stderr.log")
$ExitCode = $LASTEXITCODE
$Timer.Stop()
($RawStdout | Out-String).Trim() |
    Set-Content -LiteralPath (Join-Path $OutputPath "stdout.log") -Encoding utf8

$Result = $null
try {
    $Result = (($RawStdout | Out-String).Trim() | ConvertFrom-Json)
    if ($null -eq $Result) { throw "WSL catalog gate returned no JSON result." }
} catch {
    $Result = [pscustomobject]@{
        gate_passed = $false
        parse_error = $_.Exception.Message
        available_items = $null
        failure_count = $null
        items_per_second = $null
        peak_process_ram_mib = $null
    }
}
$Passed = $ExitCode -eq 0 -and $Result.gate_passed -eq $true
$GitCommit = (& git -C $RepositoryPath rev-parse HEAD 2>$null | Out-String).Trim()
$GitDirty = -not [string]::IsNullOrWhiteSpace((& git -C $RepositoryPath status --short 2>$null | Out-String))
$Completed = [DateTimeOffset]::UtcNow
$Receipt = [ordered]@{
    schema_version = "1.0"
    gate = "20-item-smoke"
    passed = $Passed
    exit_code = $ExitCode
    started_utc = $Started.ToString("o")
    completed_utc = $Completed.ToString("o")
    wall_seconds = $Timer.Elapsed.TotalSeconds
    git_commit = $GitCommit
    git_dirty = $GitDirty
    distribution = $Distribution
    wsl_kernel = $WslKernel
    input_aggregate_sha256 = $InputAggregateSha256
    repository_aggregate_sha256 = $RepositoryAggregateSha256
    library = $LibraryPath
    output = $OutputPath
    result = $Result
}
$ReceiptPath = Join-Path $OutputPath "smoke-receipt.json"
$Receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
@(
    "# 20-song smoke gate"
    ""
    "- Passed: $Passed"
    "- Start UTC: $($Started.ToString('o'))"
    "- End UTC: $($Completed.ToString('o'))"
    "- Wall seconds: $($Timer.Elapsed.TotalSeconds)"
    "- Git commit: $GitCommit"
    "- Git dirty: $GitDirty"
    "- WSL distribution: $Distribution"
    "- WSL kernel: $WslKernel"
    "- Input aggregate SHA-256: $InputAggregateSha256"
    "- Repository aggregate SHA-256: $RepositoryAggregateSha256"
    "- Expected/scanned: 20/$($Result.available_items)"
    "- Failures: $($Result.failure_count)"
    "- Throughput items/s: $($Result.items_per_second)"
    "- Peak process RAM MiB: $($Result.peak_process_ram_mib)"
    "- Peak VRAM MiB: unknown (catalog-only CPU gate)"
    "- Exact argv: command.json"
    "- Logs: stdout.log, stderr.log"
) | Set-Content -LiteralPath (Join-Path $OutputPath "run_summary.md") -Encoding utf8

if (-not $Passed) {
    throw "20-item smoke gate failed; see $ReceiptPath"
}
$Receipt | ConvertTo-Json -Depth 10
