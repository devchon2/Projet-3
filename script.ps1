# Récupération du chemin du dossier contenant le script PowerShell
$dossier_script = Split-Path -Parent $MyInvocation.MyCommand.Path

# Recherche du chemin de l'interpréteur Python 3.10
$python_path = Get-ChildItem -Path "C:\Program Files\Python" -Recurse -Directory -Filter "3.10.*" | Select-Object -First 1
if ($python_path -eq $null) {
    Write-Host "Erreur : l'interpréteur Python 3.10 n'a pas été trouvé."
    return
}

# Ajout du chemin de l'interpréteur Python 3.10 au PATH système
$env:Path = "$python_path;$env:Path"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)

# Création du fichier de configuration pour l'environnement Python 3.10 dans Visual Studio Code
$config = @{
    "python.pythonPath" = "$python_path\python.exe"
} | ConvertTo-Json
New-Item -ItemType File -Path "$dossier_script\.vscode\settings.json" -Force | Out-Null
Set-Content -Path "$dossier_script\.vscode\settings.json" -Value $config

Write-Host "L'environnement Python 3.10 a été configuré avec succès"