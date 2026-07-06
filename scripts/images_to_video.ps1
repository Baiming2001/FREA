param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [string]$OutputFile = "output.mp4",

    [int]$Framerate = 25,

    [string[]]$Extensions = @("*.png", "*.jpg", "*.jpeg", "*.bmp"),

    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg is not installed or not available in PATH. Please install ffmpeg first."
}

$resolvedInputDir = Resolve-FullPath -Path $InputDir

if (-not (Test-Path -LiteralPath $resolvedInputDir -PathType Container)) {
    throw "Input directory does not exist: $resolvedInputDir"
}

$images = foreach ($pattern in $Extensions) {
    Get-ChildItem -LiteralPath $resolvedInputDir -File -Filter $pattern
}

$images = $images |
    Sort-Object FullName -Unique |
    Sort-Object Name

if (-not $images -or $images.Count -eq 0) {
    throw "No supported images were found in: $resolvedInputDir"
}

$resolvedOutputFile = if ([System.IO.Path]::IsPathRooted($OutputFile)) {
    Resolve-FullPath -Path $OutputFile
} else {
    Resolve-FullPath -Path (Join-Path $resolvedInputDir $OutputFile)
}

$outputDir = Split-Path -Parent $resolvedOutputFile
if (-not (Test-Path -LiteralPath $outputDir -PathType Container)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

if ((Test-Path -LiteralPath $resolvedOutputFile) -and (-not $Overwrite)) {
    throw "Output file already exists: $resolvedOutputFile`nUse -Overwrite to replace it."
}

$listFile = Join-Path ([System.IO.Path]::GetTempPath()) ("ffmpeg_images_{0}.txt" -f [System.Guid]::NewGuid().ToString("N"))

try {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $listContent = foreach ($image in $images) {
        $escapedPath = $image.FullName.Replace("'", "''")
        "file '$escapedPath'"
    }
    [System.IO.File]::WriteAllLines($listFile, $listContent, $utf8NoBom)

    $ffmpegArgs = @(
        "-y"
        "-r", $Framerate
        "-f", "concat"
        "-safe", "0"
        "-i", $listFile
        "-vf", "fps=$Framerate,format=yuv420p"
        "-c:v", "libx264"
        $resolvedOutputFile
    )

    if (-not $Overwrite) {
        $ffmpegArgs = @(
            "-n"
            "-r", $Framerate
            "-f", "concat"
            "-safe", "0"
            "-i", $listFile
            "-vf", "fps=$Framerate,format=yuv420p"
            "-c:v", "libx264"
            $resolvedOutputFile
        )
    }

    Write-Host "Input directory: $resolvedInputDir"
    Write-Host "Image count: $($images.Count)"
    Write-Host "Output file: $resolvedOutputFile"
    Write-Host "Frame rate: $Framerate FPS"

    & ffmpeg @ffmpegArgs

    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg failed with exit code: $LASTEXITCODE"
    }

    Write-Host "Video generation completed."
}
finally {
    if (Test-Path -LiteralPath $listFile) {
        Remove-Item -LiteralPath $listFile -Force
    }
}
