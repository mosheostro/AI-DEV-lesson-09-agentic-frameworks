# Windows bootstrap for setup.sh — copy this file as `setup.ps1` next to each
# lab's `setup.sh`. It exists for the student who instinctively starts in
# PowerShell, where `source setup.sh` only produces a cryptic parse error.
#
# The real setup lives in setup.sh (bash). On Windows that runs under Git Bash,
# which ships with Git for Windows. This shim finds Git Bash and hands off to
# setup.sh inside it — or, if Git Bash isn't installed, tells you how to get it.
#
# USAGE (from this folder, in PowerShell):
#     .\setup.ps1
#
# If you already use Git Bash or WSL, ignore this file — just run:
#     source setup.sh
#
# Note: if PowerShell blocks this with an execution-policy error, the simplest
# fix is to skip it entirely and run `source setup.sh` from Git Bash directly
# (open it from the Start menu, or set it as VS Code's integrated terminal).

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- Locate Git Bash --------------------------------------------------------
# Prefer the bash.exe that ships alongside git.exe on PATH; fall back to the
# default install locations.
$bash = $null
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if ($gitCmd) {
    $gitRoot = Split-Path -Parent (Split-Path -Parent $gitCmd.Source)
    $candidate = Join-Path $gitRoot 'bin\bash.exe'
    if (Test-Path $candidate) { $bash = $candidate }
}
if (-not $bash) {
    foreach ($p in @("$env:ProgramFiles\Git\bin\bash.exe",
                     "${env:ProgramFiles(x86)}\Git\bin\bash.exe",
                     "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe")) {
        if (Test-Path $p) { $bash = $p; break }
    }
}

# --- Not found: tell the student how to get it ------------------------------
if (-not $bash) {
    Write-Host ""
    Write-Host "X  Git Bash was not found." -ForegroundColor Red
    Write-Host "   This course's setup script is bash; on Windows it runs under Git Bash."
    Write-Host ""
    Write-Host "   ACTION REQUIRED - pick one:"
    Write-Host "     1. Install Git for Windows (includes Git Bash):"
    Write-Host "          https://git-scm.com/download/win"
    Write-Host "        Then re-run:  .\setup.ps1"
    Write-Host "     2. Or use WSL (a real Linux inside Windows):"
    Write-Host "          https://learn.microsoft.com/windows/wsl/install"
    Write-Host "        Then, inside WSL:  source setup.sh"
    Write-Host ""
    Write-Host "   Tip: in VS Code, set the integrated terminal to Git Bash"
    Write-Host "        (Ctrl+Shift+P -> 'Terminal: Select Default Profile' -> Git Bash)"
    Write-Host "        and just run:  source setup.sh"
    exit 1
}

# --- Found: hand off to setup.sh inside Git Bash ----------------------------
Write-Host "Git Bash found: $bash" -ForegroundColor Green
Write-Host "Handing off to setup.sh ..." -ForegroundColor Green
Write-Host ""

# Source setup.sh (so its venv activates), then `exec bash` to drop you into an
# interactive Git Bash that inherits the activated venv. From there you're in
# bash for the rest of the lab: run exercises with  python <exercise>.py
# Forward slashes so Git Bash accepts the drive path; single quotes tolerate spaces.
$bashDir = $scriptDir -replace '\\', '/'
& $bash -lc "cd '$bashDir' && source ./setup.sh; exec bash"
