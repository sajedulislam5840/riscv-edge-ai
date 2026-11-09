# =============================================================================
# run_sim.ps1 — SIMD MAC Simulation Runner (PowerShell / Windows)
# =============================================================================
# Usage:
#   cd <repo_root>
#   .\hardware\scripts\run_sim.ps1
#
# Returns exit code 0 on all-tests-pass, 1 on failure.
# Checks for iverilog; falls back to C++ co-simulation runner if not found.
# =============================================================================

param(
    [string]$Simulator = "auto",   # "iverilog", "cpp", or "auto"
    [switch]$Verbose
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RtlDir   = Join-Path $RepoRoot "hardware\rtl"
$TbDir    = Join-Path $RepoRoot "hardware\tb"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " SIMD MAC Simulation Runner (PowerShell)  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Repo root : $RepoRoot"
Write-Host "Simulator : $Simulator"
Write-Host ""

# ---------------------------------------------------------------------------
# Detect available simulators
# ---------------------------------------------------------------------------
$HasIverilog  = $null -ne (Get-Command "iverilog"  -ErrorAction SilentlyContinue)
$HasVerilator = $null -ne (Get-Command "verilator"  -ErrorAction SilentlyContinue)
$HasGxx       = $null -ne (Get-Command "C:\msys64\ucrt64\bin\g++.exe" -ErrorAction SilentlyContinue)

Write-Host "Tool detection:"
Write-Host "  iverilog  : $(if ($HasIverilog)  { 'FOUND' } else { 'NOT FOUND' })"
Write-Host "  verilator : $(if ($HasVerilator) { 'FOUND' } else { 'NOT FOUND' })"
Write-Host "  g++ (MSYS2): $(if ($HasGxx)      { 'FOUND' } else { 'NOT FOUND' })"
Write-Host ""

# ---------------------------------------------------------------------------
# Run iverilog if available
# ---------------------------------------------------------------------------
if ($Simulator -eq "iverilog" -or ($Simulator -eq "auto" -and $HasIverilog)) {
    Write-Host "--- Running via Icarus Verilog (iverilog) ---" -ForegroundColor Yellow
    $Sources = @(
        (Join-Path $RtlDir "simd_mac.v"),
        (Join-Path $TbDir  "tb_simd_mac.v")
    )
    $OutExe = Join-Path $env:TEMP "simd_mac_sim.vvp"
    & iverilog -g2001 -Wall -o $OutExe $Sources
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] iverilog compilation failed." -ForegroundColor Red
        exit 1
    }
    & vvp $OutExe
    $ExitCode = $LASTEXITCODE
    Remove-Item -Force $OutExe -ErrorAction SilentlyContinue
    if ($ExitCode -eq 0) {
        Write-Host "[SUCCESS] iverilog simulation passed." -ForegroundColor Green
    } else {
        Write-Host "[FAILURE] iverilog simulation failed." -ForegroundColor Red
    }
    exit $ExitCode
}

# ---------------------------------------------------------------------------
# Fall back to C++ golden model verification
# ---------------------------------------------------------------------------
Write-Host "--- Falling back to C++ golden model co-simulation ---" -ForegroundColor Yellow
Write-Host "(iverilog not found; using bit-exact software model)"
Write-Host ""

$CppSrc = Join-Path $TbDir "sim_simd_mac.cpp"
$CppExe = Join-Path $env:TEMP "sim_simd_mac.exe"

if (-not (Test-Path $CppSrc)) {
    Write-Host "[ERROR] C++ co-sim source not found: $CppSrc" -ForegroundColor Red
    Write-Host "        Please run from repo root or check hardware/tb/ directory." -ForegroundColor Red
    exit 1
}

& "C:\msys64\ucrt64\bin\g++.exe" -std=c++17 -Wall -Wextra -O2 -o $CppExe $CppSrc
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] C++ compilation failed." -ForegroundColor Red
    exit 1
}

& $CppExe
$ExitCode = $LASTEXITCODE
Remove-Item -Force $CppExe -ErrorAction SilentlyContinue

if ($ExitCode -eq 0) {
    Write-Host "" 
    Write-Host "[SUCCESS] All tests passed." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[FAILURE] Tests failed — check output above." -ForegroundColor Red
}
exit $ExitCode
