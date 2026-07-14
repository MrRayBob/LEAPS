#define MyAppName "LEAPS"
#ifndef MyAppVersion
  #define MyAppVersion "2.0.0"
#endif
#define MyAppPublisher "LEAPS contributors"
#ifndef SourceDir
  #define SourceDir "..\dist\LEAPS.dist"
#endif

[Setup]
AppId={{D56F2D1A-1E0D-4D32-A4F1-4B5554B71ED1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\LEAPS
DefaultGroupName=LEAPS
DisableProgramGroupPage=yes
OutputDir=..\artifacts
OutputBaseFilename=LEAPS-Windows-x64-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayIcon={app}\LEAPS.exe

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\LEAPS"; Filename: "{app}\LEAPS.exe"
Name: "{autodesktop}\LEAPS"; Filename: "{app}\LEAPS.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\LEAPS.exe"; Description: "Launch LEAPS"; Flags: nowait postinstall skipifsilent
