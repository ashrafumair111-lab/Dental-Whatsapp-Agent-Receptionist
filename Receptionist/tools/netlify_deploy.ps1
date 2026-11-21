# Deploy the webhook site via the Netlify API (no CLI download, PS 5.1-safe).
# Requires env: NETLIFY_AUTH_TOKEN

$ErrorActionPreference = 'Stop'
$token = $env:NETLIFY_AUTH_TOKEN
if (-not $token) { throw 'NETLIFY_AUTH_TOKEN not set' }

$siteId = '6807261b-96fc-4dcb-9a70-c70aec2beb50'
$repoRoot = Split-Path $PSScriptRoot -Parent            # ...\Receptionist
$webhook = Join-Path $repoRoot 'webhook'
$tmp = Join-Path $env:TEMP 'opencode\netlify-deploy'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$headers = @{ Authorization = "Bearer $token"; Accept = 'application/json' }

function Get-EnvVar([string]$name) {
    foreach ($line in (Get-Content (Join-Path $repoRoot '.env.local'))) {
        if ($line -match "^$([regex]::Escape($name))=(.*)$") { return $matches[1] }
    }
    return $null
}

# ---- 1. Messenger env vars must already exist on the site (set in the UI) ----
$pageId = Get-EnvVar 'MESSENGER_PAGE_ID'
if (-not $pageId) { $pageId = Get-EnvVar 'messenger_page_id' }
$want = @(
    'MESSENGER_VERIFY_TOKEN',
    'MESSENGER_PAGE_ACCESS_TOKEN',
    'MESSENGER_APP_SECRET',
    'MESSENGER_PAGE_ID'
)
foreach ($k in $want) {
    $local = Get-EnvVar $k
    if (-not $local) { Write-Host "MISSING local value for $k"; exit 1 }
}

$existing = Invoke-RestMethod -Uri "https://api.netlify.com/api/v1/sites/$siteId/env" -Headers $headers
$existingKeys = @{}
foreach ($e in $existing) { $existingKeys[$e.key] = $true }

$missing = @($want | Where-Object { -not $existingKeys.ContainsKey($_) })
if ($missing.Count -gt 0) {
    Write-Host ("MISSING on site: " + ($missing -join ', ') + " - add in Netlify UI, then re-run")
    exit 1
}
Write-Host ("env OK on site: " + ($want -join ', '))

# ---- 2. Zip the deployable site (functions bundle server-side; no node_modules) ----
$stage = Join-Path $tmp 'stage'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item (Join-Path $webhook 'netlify.toml') $stage
Copy-Item (Join-Path $webhook 'package.json') $stage
Copy-Item (Join-Path $webhook 'package-lock.json') $stage
Copy-Item (Join-Path $webhook 'functions') (Join-Path $stage 'functions') -Recurse
Copy-Item (Join-Path $webhook 'public') (Join-Path $stage 'public') -Recurse

$zip = Join-Path $tmp 'deploy.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zip)
Write-Host ("zip bytes: " + (Get-Item $zip).Length)

# ---- 3. Upload as a new production deploy (manual multipart for PS 5.1) ----
$zipBytes = [System.IO.File]::ReadAllBytes($zip)
$boundary = [System.Guid]::NewGuid().ToString()
$mem = New-Object System.IO.MemoryStream
$w = New-Object System.IO.StreamWriter($mem)
$w.Write("--$boundary`r`n")
$w.Write("Content-Disposition: form-data; name=`"file`"; filename=`"deploy.zip`"`r`n")
$w.Write("Content-Type: application/zip`r`n`r`n")
$w.Flush()
$mem.Write($zipBytes, 0, $zipBytes.Length)
$w.Write("`r`n--$boundary--`r`n")
$w.Flush()
$body = $mem.ToArray()

$result = Invoke-RestMethod -Method Post -Uri "https://api.netlify.com/api/v1/sites/$siteId/deploys" `
    -Headers @{ Authorization = "Bearer $token" } `
    -ContentType "multipart/form-data; boundary=$boundary" -Body $body
Write-Host ("deploy id: " + $result.id)
Write-Host ("state: " + $result.state)
Write-Host ("ssl_url: " + $result.ssl_url)

# ---- 4. Wait until processed ----
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 15
    $d = Invoke-RestMethod -Uri "https://api.netlify.com/api/v1/deploys/$($result.id)" -Headers $headers
    Write-Host ("[{0}] state={1} error={2}" -f $i, $d.state, $d.error_message)
    if ($d.state -eq 'ready') { break }
    if ($d.state -in @('error', 'failed')) { Write-Host ("DEPLOY FAILED: " + $d.error_message); exit 1 }
}

$s = Invoke-RestMethod -Uri "https://api.netlify.com/api/v1/sites/$siteId" -Headers $headers
Write-Host ("SITE_URL=" + $s.ssl_url)
Write-Host ("MESSENGER_CALLBACK=" + $s.ssl_url + '/.netlify/functions/messenger')
