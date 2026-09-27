[CmdletBinding()]
param([ValidateRange(1,8)][int]$Jobs=4)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = $PSScriptRoot
$source = Join-Path $root 'source'
$build = Join-Path $root 'tests-build'
$logs = Join-Path $root 'source-gates-logs'
$cuda = 'E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit'
$dev = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'
$ninja = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$cl = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe'
$revision = '6da1f667e86558b152ab128edf3ebf77a80a9e57'
$targets = @(
    'strata-device',
    'dequant_s2_parity',
    's2_gemv_parity',
    'shared_expert_parity',
    'gr_parity',
    'gdn_parity',
    's2_gemv_q8_parity',
    'sampler_parity',
    'rope_parity',
    'quantize_act_parity',
    'router_top10_parity',
    's_gemv_parity',
    'elementwise_parity',
    'bf16_gemv_parity',
    's_gemv_q8k_parity',
    'qsa_parity',
    'ple_reader_test',
    'kv_q8_parity'
)

foreach ($path in @($source, $cuda, $dev, $ninja, $cl)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
}
if ((& git -C $source rev-parse HEAD).Trim() -ne $revision) { throw 'Unexpected Strata revision' }
$tracked = (& git -C $source status --porcelain --untracked-files=no) -join "`n"
if ($tracked.Trim()) { throw "Official source has tracked changes:`n$tracked" }

New-Item -ItemType Directory -Force -Path $build, $logs | Out-Null
$log = Join-Path $logs 'build-components.log'
Start-Transcript -Path $log -Force | Out-Null
try {
    $lines = & $env:ComSpec /d /s /c "`"$dev`" -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE) { throw 'Visual Studio environment failed' }
    $devPath = $null
    foreach ($line in $lines) {
        $i = $line.IndexOf('=')
        if ($i -le 0) { continue }
        $name = $line.Substring(0, $i)
        $value = $line.Substring($i + 1)
        if ($name.Equals('PATH', [StringComparison]::OrdinalIgnoreCase)) {
            if ($value -match '(?i)\\VC\\Tools\\MSVC\\[^;]+\\bin\\HostX64\\x64') { $devPath = $value }
            elseif (-not $devPath) { $devPath = $value }
        } else {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    if (-not $devPath) { throw 'Compiler PATH missing' }
    $env:PATH = "$(Join-Path $cuda 'bin');$devPath"
    $env:CUDA_PATH = $cuda

    & cmake -S $source -B $build -G Ninja `
        "-DCMAKE_MAKE_PROGRAM=$ninja" `
        "-DCMAKE_CXX_COMPILER=$cl" `
        "-DCMAKE_C_COMPILER=$cl" `
        '-DCMAKE_BUILD_TYPE=Release' `
        '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY' `
        '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release' `
        '-DSTRATA_ENABLE_CUDA=ON' `
        '-DSTRATA_BUILD_TESTS=OFF' `
        '-DSTRATA_PORTABLE=ON' `
        '-DCMAKE_CUDA_ARCHITECTURES=120' `
        "-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe" `
        "-DCUDAToolkit_ROOT=$cuda"
    if ($LASTEXITCODE) { throw 'CMake configure failed' }

    # Build only source-backed bounded components. Do not build `all`: the
    # published checkout intentionally omits tests/ and bench/micro/.
    & cmake --build $build --target @targets --parallel $Jobs
    if ($LASTEXITCODE) { throw 'Explicit component build failed' }
} finally {
    Stop-Transcript | Out-Null
}

Write-Host "Built $($targets.Count) bounded component targets in $build"
Write-Host "Build log: $log"
