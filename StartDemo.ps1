$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $Root "backend"
$FrontendDist = Join-Path $Root "frontend\dist"
$LocalUrl = "http://127.0.0.1:8000/"
$TunnelTarget = "http://127.0.0.1:8000"
$CloudflaredLocal = Join-Path $Root "cloudflared.exe"
$CloudOut = Join-Path $Root "cloudflared.out.log"
$CloudErr = Join-Path $Root "cloudflared.err.log"
$PhoneQrHtml = Join-Path $Root "phone-url-qr.html"

function Write-Step($Message) {
  Write-Host ""
  Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Wait-For-Backend {
  param([int]$TimeoutSeconds = 20)
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
      if ($response.StatusCode -eq 200) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  return $false
}

function Find-Python {
  $venvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
  if (Test-Path -LiteralPath $venvPython) {
    return $venvPython
  }
  $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($pythonCommand) {
    return $pythonCommand.Source
  }
  $pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
  if ($pyCommand) {
    return $pyCommand.Source
  }
  return $null
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

function Ensure-PythonDependencies {
  param([string]$PythonPath)
  Write-Step "Checking Python dependencies"
  $check = & $PythonPath -c "import numpy, pandas, matplotlib, sklearn" 2>$null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "Python dependencies are available." -ForegroundColor Green
    return
  }

  $requirements = Join-Path $BackendDir "requirements.txt"
  Write-Host "Python dependencies are missing. Installing from $requirements ..." -ForegroundColor Yellow
  & $PythonPath -m pip install -r $requirements
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install Python dependencies." -ForegroundColor Red
    Write-Host "Install them manually with: python -m pip install -r backend\requirements.txt"
    Read-Host "Press Enter to close"
    exit 1
  }
}

Write-Host "Motion Recognition Prototype Demo Launcher" -ForegroundColor Green
Write-Host "Root: $Root"

if (-not (Test-Path -LiteralPath $FrontendDist)) {
  Write-Host "frontend/dist was not found." -ForegroundColor Red
  Write-Host "Run this on the development PC first: cd frontend; npm.cmd run build"
  Read-Host "Press Enter to close"
  exit 1
}

$Python = Find-Python
if (-not $Python) {
  Write-Host "Python was not found. Install Python 3.10+ or use a package that includes StartDemo.exe." -ForegroundColor Red
  Read-Host "Press Enter to close"
  exit 1
}

$Cloudflared = Find-Cloudflared
if (-not $Cloudflared) {
  Write-Host "cloudflared.exe was not found." -ForegroundColor Red
  Write-Host "Place cloudflared.exe next to StartDemo.bat, or install cloudflared and add it to PATH."
  Read-Host "Press Enter to close"
  exit 1
}

Ensure-PythonDependencies -PythonPath $Python

Write-Step "Starting backend"
$backendCommand = "cd /d `"$BackendDir`" && `"$Python`" server.py"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $backendCommand -WindowStyle Normal

if (-not (Wait-For-Backend -TimeoutSeconds 25)) {
  Write-Host "Backend did not respond on http://127.0.0.1:8000." -ForegroundColor Red
  Write-Host "If port 8000 is already in use, close the other app and run StartDemo again."
  Read-Host "Press Enter to close"
  exit 1
}

Write-Host "Backend is running: $LocalUrl" -ForegroundColor Green

Write-Step "Opening PC Battle screen"
Start-Process $LocalUrl

Write-Step "Starting cloudflared tunnel"
Remove-Item -LiteralPath $CloudOut, $CloudErr -Force -ErrorAction SilentlyContinue
$cloudProcess = Start-Process `
  -FilePath $Cloudflared `
  -ArgumentList @("tunnel", "--url", $TunnelTarget) `
  -RedirectStandardOutput $CloudOut `
  -RedirectStandardError $CloudErr `
  -WindowStyle Hidden `
  -PassThru

$phoneUrl = $null
$deadline = (Get-Date).AddSeconds(45)
while ((Get-Date) -lt $deadline -and -not $phoneUrl) {
  Start-Sleep -Milliseconds 700
  $text = ""
  if (Test-Path -LiteralPath $CloudOut) {
    $text += Get-Content -LiteralPath $CloudOut -Raw -ErrorAction SilentlyContinue
  }
  if (Test-Path -LiteralPath $CloudErr) {
    $text += "`n" + (Get-Content -LiteralPath $CloudErr -Raw -ErrorAction SilentlyContinue)
  }
  $match = [regex]::Match($text, "https://[a-zA-Z0-9-]+\.trycloudflare\.com")
  if ($match.Success) {
    $phoneUrl = $match.Value + "/?mode=phone"
  }
}

Write-Host ""
if ($phoneUrl) {
  $encodedPhoneUrl = [uri]::EscapeDataString($phoneUrl)
  $qrImageUrl = "https://api.qrserver.com/v1/create-qr-code/?size=420x420&data=$encodedPhoneUrl"
  $html = @"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M1GP QUEST Phone URL</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #0b1020;
      color: #fff;
      font-family: Consolas, "Courier New", monospace;
    }
    main {
      width: min(760px, calc(100vw - 32px));
      padding: 28px;
      border: 2px solid #38bdf8;
      border-radius: 18px;
      background: #111827;
      text-align: center;
      box-shadow: 0 20px 60px rgba(0,0,0,.45);
    }
    h1 { margin: 0 0 18px; font-size: 28px; }
    img { width: min(420px, 82vw); height: auto; background: #fff; padding: 16px; border-radius: 14px; }
    p { overflow-wrap: anywhere; line-height: 1.6; }
    .url { color: #86efac; font-size: 18px; }
  </style>
</head>
<body>
  <main>
    <h1>M1GP QUEST Phone URL</h1>
    <img src="$qrImageUrl" alt="Phone QR code">
    <p>Scan this QR code with the phone.</p>
    <p class="url">$phoneUrl</p>
  </main>
</body>
</html>
"@
  Set-Content -LiteralPath $PhoneQrHtml -Value $html -Encoding UTF8
  Start-Process $PhoneQrHtml
  Write-Host "========================================" -ForegroundColor Yellow
  Write-Host "PHONE URL" -ForegroundColor Yellow
  Write-Host $phoneUrl -ForegroundColor Green
  Write-Host "QR code page: $PhoneQrHtml" -ForegroundColor Green
  Write-Host "========================================" -ForegroundColor Yellow
  try {
    Set-Clipboard -Value $phoneUrl
    Write-Host "Phone URL copied to clipboard."
  } catch {
    Write-Host "Could not copy URL to clipboard."
  }
} else {
  Write-Host "Could not detect the cloudflared URL automatically." -ForegroundColor Red
  Write-Host "Check cloudflared.err.log for a https://xxxxx.trycloudflare.com URL."
  if (Test-Path -LiteralPath $CloudErr) {
    Write-Host ""
    Get-Content -LiteralPath $CloudErr -Tail 20
  }
}

Write-Host ""
Write-Host "PC URL: $LocalUrl"
Write-Host "Keep this window open while using the phone tunnel."
Write-Host "Press Enter here to stop cloudflared. Close the backend window when finished."
Read-Host

if ($cloudProcess -and -not $cloudProcess.HasExited) {
  Stop-Process -Id $cloudProcess.Id -Force
}
