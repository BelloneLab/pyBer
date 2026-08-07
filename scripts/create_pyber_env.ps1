param(
    [switch]$Recreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    <#
    Returns the repository root based on this script's location. Keeping path
    discovery local to the script lets users run it from any working directory.
    #>
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Test-CondaEnvExists {
    <#
    Checks whether the named conda environment exists by parsing Conda's JSON
    output. This is more reliable than parsing localized table output.
    #>
    param([Parameter(Mandatory = $true)][string]$Name)

    $json = conda env list --json | ConvertFrom-Json
    foreach ($envPath in $json.envs) {
        if ((Split-Path $envPath -Leaf) -eq $Name) {
            return $true
        }
    }
    return $false
}

function Invoke-Conda {
    <#
    Runs a conda command and fails immediately if Conda returns a non-zero exit
    code. The wrapper keeps error handling explicit and easy to audit.
    #>
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & conda @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "conda $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$repoRoot = Get-RepoRoot
$environmentFile = Join-Path $repoRoot "environment.yml"
$fastFmmScript = Join-Path $repoRoot "scripts\install_fastfmm.R"
$envName = "pyBer"

if ($Recreate -and (Test-CondaEnvExists -Name $envName)) {
    Invoke-Conda -Arguments @("env", "remove", "-n", $envName, "-y")
}

if (Test-CondaEnvExists -Name $envName) {
    Invoke-Conda -Arguments @("env", "update", "-n", $envName, "-f", $environmentFile, "--prune")
} else {
    Invoke-Conda -Arguments @("env", "create", "-f", $environmentFile)
}

# Run the R installer inside the conda environment so fastFMM is installed into
# the R library that pyBer will use through rpy2.
Invoke-Conda -Arguments @("run", "-n", $envName, "Rscript", $fastFmmScript)

Write-Host "pyBer environment is ready."
Write-Host "Activate it with: conda activate pyBer"
Write-Host "Launch with: python .\pyBer\main.py"
