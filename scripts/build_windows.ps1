$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$ProjectVersion = [string](python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the LEAPS version from pyproject.toml"
}
$Version = if ($env:LEAPS_VERSION) { $env:LEAPS_VERSION.Trim() } else { $ProjectVersion.Trim() }
if ($Version.StartsWith("v")) {
    $Version = $Version.Substring(1)
}
if ($Version -notmatch '^\d+(\.\d+){1,2}$') {
    throw "LEAPS_VERSION must contain two or three numeric components (received: $Version)"
}
$env:LEAPS_VERSION = $Version

python -m pip install --upgrade "PyInstaller>=6.14,<7"
python -m PyInstaller --noconfirm --clean packaging/LEAPS-windows.spec

$DeploymentDir = Get-Item -Path "dist/LEAPS" -ErrorAction SilentlyContinue
if (!$DeploymentDir) {
    throw "A standalone LEAPS deployment directory was not produced under dist"
}
$Executable = Get-Item -Path (Join-Path $DeploymentDir.FullName "LEAPS.exe") -ErrorAction SilentlyContinue
if (!$Executable) {
    throw "The LEAPS application executable was not produced in $($DeploymentDir.FullName)"
}

& $Executable.FullName --packaging-self-test
if ($LASTEXITCODE -ne 0) {
    throw "The packaged LEAPS runtime self-test failed"
}

New-Item -ItemType Directory -Force -Path artifacts | Out-Null
$SourceDir = $Executable.DirectoryName
if ($env:WINDOWS_CERTIFICATE_PATH) {
    signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
        /f $env:WINDOWS_CERTIFICATE_PATH /p $env:WINDOWS_CERTIFICATE_PASSWORD `
        $Executable.FullName
    signtool verify /pa /v $Executable.FullName
}
$IsccCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($IsccCommand) {
    $IsccPath = $IsccCommand.Source
} else {
    $IsccPath = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6/ISCC.exe"
    if (!(Test-Path $IsccPath)) {
        throw "Inno Setup 6 was not found on the Windows runner"
    }
}
& $IsccPath "/DSourceDir=$SourceDir" "/DMyAppVersion=$Version" packaging/leaps.iss
$Installer = "artifacts/LEAPS-Windows-x64-Setup.exe"
if (!(Test-Path $Installer)) {
    throw "The Windows installer was not produced"
}

if ($env:WINDOWS_CERTIFICATE_PATH) {
    signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
        /f $env:WINDOWS_CERTIFICATE_PATH /p $env:WINDOWS_CERTIFICATE_PASSWORD `
        $Installer
    signtool verify /pa /v $Installer
}
