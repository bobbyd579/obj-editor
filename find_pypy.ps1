# Script to find PyPy installation
Write-Host "Searching for PyPy installation..." -ForegroundColor Cyan
Write-Host ""

# Common locations to check
$locations = @(
    "C:\pypy*",
    "C:\Program Files\pypy*",
    "C:\Program Files (x86)\pypy*",
    "$env:LOCALAPPDATA\Programs\pypy*",
    "$env:USERPROFILE\AppData\Local\Programs\pypy*"
)

$found = $false

foreach ($location in $locations) {
    $dirs = Get-ChildItem -Path $location -Directory -ErrorAction SilentlyContinue
    foreach ($dir in $dirs) {
        $pypyExe = Join-Path $dir.FullName "pypy.exe"
        if (Test-Path $pypyExe) {
            Write-Host "Found PyPy at: $($dir.FullName)" -ForegroundColor Green
            Write-Host "  Executable: $pypyExe" -ForegroundColor Yellow
            $version = & $pypyExe --version 2>&1
            Write-Host "  Version: $version" -ForegroundColor Yellow
            Write-Host ""
            $found = $true
        }
    }
}

if (-not $found) {
    Write-Host "PyPy not found in common locations." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please find PyPy manually:" -ForegroundColor Yellow
    Write-Host "1. Open File Explorer" -ForegroundColor White
    Write-Host "2. Go to C:\" -ForegroundColor White
    Write-Host "3. Search for 'pypy.exe'" -ForegroundColor White
    Write-Host "4. Note the full path (e.g., C:\pypy3.10\pypy.exe)" -ForegroundColor White
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

