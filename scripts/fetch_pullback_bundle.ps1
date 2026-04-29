param(
    [Parameter(Mandatory = $true)]
    [string]$Host,

    [string]$User = "ubuntu",

    [Parameter(Mandatory = $true)]
    [string]$BundleName,

    [string]$RemoteDir = "/home/ubuntu/cjm/server_train_resume_pack/pullback_bundles",

    [string]$Destination = "archive"
)

$ErrorActionPreference = "Stop"

$destRoot = Resolve-Path "." | Select-Object -ExpandProperty Path
$targetDir = Join-Path $destRoot $Destination
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$files = @(
    "$BundleName.tar.gz",
    "${BundleName}_included.txt",
    "${BundleName}_missing.txt"
)

foreach ($file in $files) {
    $remote = "$User@$Host`:$RemoteDir/$file"
    $local = Join-Path $targetDir $file
    Write-Host "Fetching $remote -> $local"
    scp $remote $local
}

Write-Host "Done. Files saved under $targetDir"
