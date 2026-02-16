[CmdletBinding()]
param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

function Resolve-PythonCommand {
    $candidates = @(
        @{ Exe = "python"; Args = @() },
        @{ Exe = "python3"; Args = @() },
        @{ Exe = "python.exe"; Args = @() },
        @{ Exe = "py"; Args = @("-3") },
        @{ Exe = "py.exe"; Args = @("-3") }
    )

    foreach ($candidate in $candidates) {
        if (Get-Command $candidate.Exe -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }

    throw "Could not find a Python interpreter (python/python3/python.exe/py)."
}

$pythonCmd = Resolve-PythonCommand
$pythonExe = $pythonCmd.Exe
$pythonArgs = @($pythonCmd.Args)

Write-Host ("Using Python interpreter: {0} {1}" -f $pythonExe, ($pythonArgs -join " "))

# Figure 3 corresponds to experiment 1:
# - 5 tasks, 5 epochs per task
# - MLP with 2 hidden layers of 100 units
# - Compare Plastic (Naive) vs Stable regimes on Rotated/Permuted MNIST
# - Compute top-20 Hessian eigenvalues for the middle row in Fig.3
$tasks = "5"
$epochsPerTask = "5"
$hiddens = "100"
$numEigenthings = "20"
$seeds = @("1234", "4567", "7891", "1145", "9723")

function Invoke-RunRegime {
    param(
        [Parameter(Mandatory = $true)][string]$Dataset,
        [Parameter(Mandatory = $true)][string]$RegimeName,
        [Parameter(Mandatory = $true)][string]$Lr,
        [Parameter(Mandatory = $true)][string]$Gamma,
        [Parameter(Mandatory = $true)][string]$BatchSize,
        [Parameter(Mandatory = $true)][string]$Dropout
    )

    Write-Host (" >>>>>>>> {0} ({1})" -f $RegimeName, $Dataset)
    foreach ($seed in $seeds) {
        Write-Host ("seed={0}, lr={1}, gamma={2}, bs={3}, dropout={4}" -f $seed, $Lr, $Gamma, $BatchSize, $Dropout)

        $args = @()
        $args += $pythonArgs
        $args += @(
            "-m", "stable_sgd.main",
            "--dataset", $Dataset,
            "--tasks", $tasks,
            "--epochs-per-task", $epochsPerTask,
            "--lr", $Lr,
            "--gamma", $Gamma,
            "--hiddens", $hiddens,
            "--batch-size", $BatchSize,
            "--dropout", $Dropout,
            "--seed", $seed,
            "--compute-eigenspectrum",
            "--num-eigenthings", $numEigenthings
        )

        if ($DryRun) {
            Write-Host ("[DryRun] {0} {1}" -f $pythonExe, ($args -join " "))
        }
        else {
            & $pythonExe @args
        }
    }
    Write-Host ""
}

Write-Host "************************ replicating Figure 3 (rotated MNIST) ***********************"
Invoke-RunRegime -Dataset "rot-mnist" -RegimeName "Plastic (Naive) SGD" -Lr "0.01" -Gamma "1.0" -BatchSize "64" -Dropout "0.0"
Invoke-RunRegime -Dataset "rot-mnist" -RegimeName "Stable SGD" -Lr "0.1" -Gamma "0.4" -BatchSize "16" -Dropout "0.25"

Write-Host "************************ replicating Figure 3 (permuted MNIST) ***********************"
Invoke-RunRegime -Dataset "perm-mnist" -RegimeName "Plastic (Naive) SGD" -Lr "0.01" -Gamma "1.0" -BatchSize "64" -Dropout "0.0"
Invoke-RunRegime -Dataset "perm-mnist" -RegimeName "Stable SGD" -Lr "0.1" -Gamma "0.4" -BatchSize "16" -Dropout "0.25"
