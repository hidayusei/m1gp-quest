$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $Root "backend"
$FrontendDir = Join-Path $Root "frontend"
$BackendPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
$CloudflaredLocal = Join-Path $Root "cloudflared.exe"
$BackendUrl = "http://127.0.0.1:8000"
$PcUrl = "http://127.0.0.1:5173/"
$TunnelTarget = "http://127.0.0.1:5173"
$CloudOut = Join-Path $Root "dev-cloudflared.out.log"
$CloudErr = Join-Path $Root "dev-cloudflared.err.log"

$BackendProcess = $null
$FrontendProcess = $null
$CloudProcess = $null

function Write-Header {
  Clear-Host
  Write-Host "========================================" -ForegroundColor Cyan
  Write-Host "       M1 QUEST Demo Launcher" -ForegroundColor Yellow
  Write-Host "========================================" -ForegroundColor Cyan
  Write-Host ""
}

function Stop-LauncherProcesses {
  foreach ($process in @($CloudProcess, $FrontendProcess, $BackendProcess)) {
    if ($null -ne $process) {
      try {
        if (-not $process.HasExited) {
          Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
      } catch {
        # A process may already have been closed from its own console window.
      }
    }
  }
}

function Fail-Launcher {
  param([string]$Message)
  Write-Host ""
  Write-Host "ERROR: $Message" -ForegroundColor Red
  Write-Host ""
  Stop-LauncherProcesses
  Read-Host "Press Enter to close"
  exit 1
}

function Test-LocalPort {
  param([int]$Port)
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $connect = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
    if (-not $connect.AsyncWaitHandle.WaitOne(300)) {
      return $false
    }
    $client.EndConnect($connect)
    return $true
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

function Wait-ForUrl {
  param(
    [string]$Url,
    [int]$TimeoutSeconds = 30
  )
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  return $false
}

function Find-Cloudflared {
  if (Test-Path -LiteralPath $CloudflaredLocal) {
    return $CloudflaredLocal
  }
  $command = Get-Command cloudflared.exe -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }
  return $null
}

Write-Header
Write-Host "Project: $Root"

if (-not (Test-Path -LiteralPath $BackendPython)) {
  Fail-Launcher "Backend Python was not found: $BackendPython"
}
if (-not (Test-Path -LiteralPath (Join-Path $BackendDir "server.py"))) {
  Fail-Launcher "backend\server.py was not found."
}
if (-not (Test-Path -LiteralPath (Join-Path $FrontendDir "package.json"))) {
  Fail-Launcher "frontend\package.json was not found."
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
  Fail-Launcher "npm.cmd was not found. Install Node.js and reopen this launcher."
}
$Cloudflared = Find-Cloudflared
if (-not $Cloudflared) {
  Fail-Launcher "cloudflared.exe was not found. Place it in the repository root or add it to PATH."
}

$occupiedPorts = @()
if (Test-LocalPort -Port 8000) { $occupiedPorts += 8000 }
if (Test-LocalPort -Port 5173) { $occupiedPorts += 5173 }
if ($occupiedPorts.Count -gt 0) {
  Fail-Launcher "Port(s) already in use: $($occupiedPorts -join ', '). Close the existing backend/frontend and run again."
}

Write-Host "[1/4] Starting backend..." -ForegroundColor Cyan
$backendCommand = "cd /d `"$BackendDir`" && `"$BackendPython`" server.py"
$BackendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $backendCommand -WindowStyle Normal -PassThru
if (-not (Wait-ForUrl -Url "$BackendUrl/health" -TimeoutSeconds 30)) {
  Fail-Launcher "Backend did not become ready at $BackendUrl/health"
}
Write-Host "      Backend ready: $BackendUrl" -ForegroundColor Green

Write-Host "[2/4] Starting frontend..." -ForegroundColor Cyan
$frontendCommand = "cd /d `"$FrontendDir`" && npm.cmd run dev -- --host 0.0.0.0 --port 5173"
$FrontendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $frontendCommand -WindowStyle Normal -PassThru
if (-not (Wait-ForUrl -Url $PcUrl -TimeoutSeconds 40)) {
  Fail-Launcher "Frontend did not become ready at $PcUrl"
}
Write-Host "      Frontend ready: $PcUrl" -ForegroundColor Green

Write-Host "[3/4] Opening PC browser..." -ForegroundColor Cyan
Start-Process $PcUrl

Write-Host "[4/4] Starting cloudflared..." -ForegroundColor Cyan
Remove-Item -LiteralPath $CloudOut, $CloudErr -Force -ErrorAction SilentlyContinue
$CloudProcess = Start-Process `
  -FilePath $Cloudflared `
  -ArgumentList @("tunnel", "--url", $TunnelTarget) `
  -RedirectStandardOutput $CloudOut `
  -RedirectStandardError $CloudErr `
  -WindowStyle Hidden `
  -PassThru

$PhoneUrl = $null
$deadline = (Get-Date).AddSeconds(45)
while ((Get-Date) -lt $deadline -and -not $PhoneUrl) {
  if ($CloudProcess.HasExited) {
    break
  }
  Start-Sleep -Milliseconds 500
  $cloudText = ""
  if (Test-Path -LiteralPath $CloudOut) {
    $cloudText += Get-Content -LiteralPath $CloudOut -Raw -ErrorAction SilentlyContinue
  }
  if (Test-Path -LiteralPath $CloudErr) {
    $cloudText += "`n" + (Get-Content -LiteralPath $CloudErr -Raw -ErrorAction SilentlyContinue)
  }
  $match = [regex]::Match($cloudText, "https://[a-zA-Z0-9-]+\.trycloudflare\.com")
  if ($match.Success) {
    $PhoneUrl = $match.Value.TrimEnd("/") + "/?mode=phone"
  }
}

if (-not $PhoneUrl) {
  Write-Host ""
  Write-Host "cloudflared output:" -ForegroundColor Yellow
  if (Test-Path -LiteralPath $CloudErr) {
    Get-Content -LiteralPath $CloudErr -Tail 20
  }
  Fail-Launcher "Could not detect a trycloudflare.com URL within 45 seconds."
}

$clipboardCopied = $false
try {
  Set-Clipboard -Value $PhoneUrl
  $clipboardCopied = $true
} catch {
  $clipboardCopied = $false
}

Write-Header
Write-Host "PC URL:" -ForegroundColor Cyan
Write-Host $PcUrl -ForegroundColor Green
Write-Host ""
Write-Host "Phone URL:" -ForegroundColor Cyan
Write-Host $PhoneUrl -ForegroundColor Yellow
Write-Host ""
if ($clipboardCopied) {
  Write-Host "The phone URL has been copied to clipboard." -ForegroundColor Green
} else {
  Write-Host "Could not copy the phone URL to clipboard." -ForegroundColor Yellow
}
Write-Host "Keep this window open." -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to stop backend, frontend, and cloudflared"
Stop-LauncherProcesses
Write-Host "Development demo stopped." -ForegroundColor Green
