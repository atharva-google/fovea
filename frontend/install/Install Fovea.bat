@echo off
rem Fovea setup for Windows. Double-click it once; run it again to update Fovea.
rem It needs nothing installed: it brings its own Python, libraries and ffmpeg, all inside one folder.
rem To remove Fovea, delete %LOCALAPPDATA%\Fovea and the Fovea shortcuts.
set "FOVEA_SELF=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = Get-Content -Raw -LiteralPath $env:FOVEA_SELF; iex $s.Substring($s.IndexOf('#' + '>') + 2)"
echo.
pause
exit /b
#>
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = if ($env:FOVEA_INSTALL_DIR) { $env:FOVEA_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Fovea' }
$Branch = if ($env:FOVEA_BRANCH) { $env:FOVEA_BRANCH } else { 'feat/frontend' }
$Repo = 'https://github.com/atharva-google/fovea'

function Say($text) { Write-Host "`n$text" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what did not work" } }

try {
    New-Item -ItemType Directory -Force $Root | Out-Null

    Say '1/5  Python tools'
    $env:UV_INSTALL_DIR = Join-Path $Root 'bin'
    $env:UV_NO_MODIFY_PATH = '1'
    $env:INSTALLER_NO_MODIFY_PATH = '1'
    $Uv = Join-Path $Root 'bin\uv.exe'
    if (!(Test-Path $Uv)) { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression }

    Say '2/5  Fovea itself'
    $Zip = Join-Path $env:TEMP 'fovea.zip'
    $Tmp = Join-Path $env:TEMP 'fovea_unzip'
    Invoke-WebRequest "$Repo/archive/refs/heads/$Branch.zip" -OutFile $Zip -UseBasicParsing
    if (Test-Path $Tmp) { Remove-Item -Recurse -Force $Tmp }
    Expand-Archive $Zip $Tmp
    if (Test-Path "$Root\app") { Remove-Item -Recurse -Force "$Root\app" }
    Move-Item (Get-ChildItem $Tmp | Select-Object -First 1).FullName "$Root\app"
    Remove-Item -Recurse -Force $Tmp, $Zip

    Say '3/5  Python and the libraries (a few minutes the first time)'
    $Python = Join-Path $Root 'venv\Scripts\python.exe'
    if (!(Test-Path $Python)) { & $Uv venv --python 3.12 "$Root\venv"; Check 'Installing Python' }
    & $Uv pip install --python $Python -r "$Root\app\requirements.txt" static-ffmpeg; Check 'Installing the libraries'

    Say '4/5  The speech model and ffmpeg'
    & $Python "$Root\app\frontend\launch.py" --prepare; Check 'Getting the speech model'

    Say '5/5  The Fovea icon'
    $Quiet = Join-Path $Root 'venv\Scripts\pythonw.exe'
    $Shell = New-Object -ComObject WScript.Shell
    foreach ($Folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
        $Link = $Shell.CreateShortcut((Join-Path $Folder 'Fovea.lnk'))
        $Link.TargetPath = $Quiet
        $Link.Arguments = '"' + "$Root\app\frontend\launch.py" + '"'
        $Link.WorkingDirectory = $Root
        $Link.Description = 'Fovea'
        $Link.Save()
    }
    Write-Host 'Fovea is on your Desktop and in the Start menu.'

    Say 'Done. Opening Fovea ...'
    Start-Process $Quiet -ArgumentList ('"' + "$Root\app\frontend\launch.py" + '"') -WorkingDirectory $Root
    Write-Host 'From now on, click the Fovea icon. You can close this window.'
} catch {
    Write-Host "`nSetup did not finish: $_" -ForegroundColor Red
    Write-Host 'Check your internet connection and run this file again.'
}
