; Inno Setup 6 Script for Lumens Deep
; Documentation: https://jrsoftware.org/ishelp/

#define MyAppName "Lumens Deep"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Volodymyr Makovetskyi"
#define MyAppURL "https://lumensdeep.com"
#define MyAppExeName "LumensDeep.exe"

[Setup]
; AppId identifies this application uniquely in Windows
AppId={{D37F8E92-6C14-4B2E-A1C8-527D114620A1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v0.1 (Alpha)
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=output
OutputBaseFilename=LumensDeep_Setup_v0.1_Alpha_x64
SetupIconFile=assets\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes

[Languages]
Name: "ukrainian"; MessagesFile: "compiler:Languages\Ukrainian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Dist folder compiled by PyInstaller (Strictly exclude any .env or sensitive credentials)
Source: "dist\LumensDeep\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.env,*.env.*,.env,*.key,*.token,*.secret,ai_error_*.txt,*.pth,*.tflite"

[Icons]
; Start Menu shortcuts (Меню "Пуск")
Name: "{autoprograms}\{#MyAppName}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\{#MyAppName}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; Desktop shortcut (Ярлик на Робочому столі)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Auto-launch prompt after setup completes
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
