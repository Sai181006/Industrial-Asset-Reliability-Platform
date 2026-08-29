param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\data\raw")
)

$ErrorActionPreference = "Stop"
$url = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
$downloadDir = Join-Path $Destination "downloads"
$sourceDir = Join-Path $Destination "source"
$outerZip = Join-Path $downloadDir "Turbofan_Engine_Degradation_Simulation_Data_Set.zip"
$outerExtract = Join-Path $sourceDir "official_archive"
$datasetDir = Join-Path $sourceDir "CMAPSSData"

New-Item -ItemType Directory -Force -Path $downloadDir, $sourceDir | Out-Null
if (-not (Test-Path -LiteralPath $outerZip)) {
    Invoke-WebRequest -Uri $url -OutFile $outerZip
}
if (-not (Test-Path -LiteralPath $datasetDir)) {
    Expand-Archive -LiteralPath $outerZip -DestinationPath $outerExtract -Force
    $innerZip = Get-ChildItem -LiteralPath $outerExtract -Recurse -File -Filter "CMAPSSData.zip" | Select-Object -First 1
    if (-not $innerZip) { throw "CMAPSSData.zip was not found in the official archive." }
    Expand-Archive -LiteralPath $innerZip.FullName -DestinationPath $datasetDir -Force
}

$required = "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt"
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $datasetDir $name))) {
        throw "Expected file missing: $name"
    }
}
Get-FileHash -Algorithm SHA256 -LiteralPath $outerZip
Write-Output "FD001 source directory: $datasetDir"

