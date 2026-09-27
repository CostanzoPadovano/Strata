[CmdletBinding()]
param([ValidateRange(1,8)][int]$Jobs=4)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$source=Join-Path $PSScriptRoot 'source'
$build=Join-Path $PSScriptRoot 'build'
$ggml=Join-Path (Split-Path $PSScriptRoot -Parent) 'qwen-strata-stock-20260926/tests-build/_deps/strata_llamacpp-src'
$cuda='E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit'
$dev='C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'
$ninja='C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$cl='C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe'
foreach($path in @($source,$ggml,$cuda,$dev,$ninja,$cl)) {
    if(-not(Test-Path -LiteralPath $path)){throw "Missing build prerequisite: $path"}
}
if((& git -C $source rev-parse HEAD).Trim() -ne '6da1f667e86558b152ab128edf3ebf77a80a9e57'){throw 'Unexpected source base'}
if((& git -C $ggml rev-parse HEAD).Trim() -ne '3cf03257f219afbe7334045ff7c6a06ac68c627d'){throw 'Unexpected ggml pin'}
New-Item -ItemType Directory -Force -Path $build | Out-Null
$sourceInputs=@()
$transportInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'transport_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant()
$registrationInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'host_registration_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant()
$embeddingInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'embedding_cache_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant()
$mtpEquivalenceInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'mtp_equivalence_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
$mtpProfileInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'mtp_profile_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
$remoteExpertsInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'remote_experts_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
$remoteExpertsProfileInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'remote_experts_profile_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
$compactedLoaderInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'compacted_loader_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant()
$verifyWaitInput=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'verify_wait_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant()
$trackedNames=@(& git -C $source ls-files)+@('include/strata/core/remote_experts.hpp','src/core/remote_experts.cpp')
foreach($relative in ($trackedNames | Sort-Object -Unique)) {
    $file=Join-Path $source $relative
    if(Test-Path -LiteralPath $file -PathType Leaf) {
        $sourceInputs+=@{path=$relative;sha256=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
}
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
Start-Transcript -Path (Join-Path $build "build-$stamp.log") -NoClobber | Out-Null
try {
    $lines=& $env:ComSpec /d /s /c "`"$dev`" -arch=x64 -host_arch=x64 >nul && set"
    if($LASTEXITCODE){throw 'Visual Studio environment failed'}
    $devPath=$null
    foreach($line in $lines){
        $i=$line.IndexOf('=');if($i -le 0){continue}
        $name=$line.Substring(0,$i);$value=$line.Substring($i+1)
        if($name.Equals('PATH',[StringComparison]::OrdinalIgnoreCase)){
            if($value -match '(?i)\\VC\\Tools\\MSVC\\[^;]+\\bin\\HostX64\\x64'){$devPath=$value}
            elseif(-not $devPath){$devPath=$value}
        } else {Set-Item -Path "Env:$name" -Value $value}
    }
    if(-not $devPath){throw 'Compiler PATH missing'}
    $env:PATH="$(Join-Path $cuda 'bin');$devPath"
    $env:CUDA_PATH=$cuda
    & cmake -S $source -B $build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" `
        "-DCMAKE_CXX_COMPILER=$cl" "-DCMAKE_C_COMPILER=$cl" `
        '-DCMAKE_BUILD_TYPE=Release' '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY' `
        '-DSTRATA_ENABLE_CUDA=ON' '-DSTRATA_BUILD_TESTS=OFF' '-DSTRATA_ADVANCED_TESTS=ON' '-DSTRATA_PORTABLE=ON' `
        '-DCMAKE_CUDA_ARCHITECTURES=120' "-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe" `
        "-DCUDAToolkit_ROOT=$cuda" "-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=$ggml"
    if($LASTEXITCODE){throw 'CMake configure failed'}
    & cmake --build $build --target strata mtp_transport_test host_registration_test embedding_cache_test mtp_equivalence_test mtp_profile_test remote_experts_test compacted_loader_test remote_experts_profile_test verify_wait_test --parallel $Jobs
    if($LASTEXITCODE){throw 'Engine build failed'}
    foreach($row in $sourceInputs) {
        if((Get-FileHash -LiteralPath (Join-Path $source $row.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
            throw "Source changed during build: $($row.path)"
        }
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'transport_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $transportInput) {
        throw 'Transport test source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'host_registration_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $registrationInput) {
        throw 'Registration test source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'embedding_cache_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $embeddingInput) {
        throw 'Embedding test source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'mtp_equivalence_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mtpEquivalenceInput) {
        throw 'MTP equivalence test source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'mtp_profile_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $mtpProfileInput) {
        throw 'MTP profile test source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'remote_experts_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $remoteExpertsInput) {
        throw 'Remote expert test source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'remote_experts_profile_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $remoteExpertsProfileInput) {
        throw 'Remote expert profile source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'compacted_loader_test.cpp') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $compactedLoaderInput) {
        throw 'Compacted loader test source changed during build'
    }
    if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'verify_wait_test.cu') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $verifyWaitInput) {
        throw 'Verifier wait test source changed during build'
    }
    $binding=@{source_files=$sourceInputs;source_base='6da1f667e86558b152ab128edf3ebf77a80a9e57';
        ggml_revision='3cf03257f219afbe7334045ff7c6a06ac68c627d';cuda='13.3.73';architecture='120';
        transport_source_sha256=$transportInput;
        registration_source_sha256=$registrationInput;
        embedding_source_sha256=$embeddingInput;
        mtp_equivalence_source_sha256=$mtpEquivalenceInput;
        mtp_profile_source_sha256=$mtpProfileInput;
        remote_experts_source_sha256=$remoteExpertsInput;
        remote_experts_profile_source_sha256=$remoteExpertsProfileInput;
        compacted_loader_source_sha256=$compactedLoaderInput;
        verify_wait_source_sha256=$verifyWaitInput;
        executable_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'strata.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        registration_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'host_registration_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        embedding_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'embedding_cache_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        mtp_equivalence_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'mtp_equivalence_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        mtp_profile_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'mtp_profile_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        remote_experts_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'remote_experts_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        remote_experts_profile_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'remote_experts_profile_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        compacted_loader_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'compacted_loader_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        verify_wait_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'verify_wait_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
        transport_sha256=(Get-FileHash -LiteralPath (Join-Path $build 'mtp_transport_test.exe') -Algorithm SHA256).Hash.ToLowerInvariant()}
    # Generated build evidence, not a source/config edit.
    [IO.File]::WriteAllText((Join-Path $build 'build-binding.json'),($binding|ConvertTo-Json -Depth 8))
} finally {Stop-Transcript | Out-Null}
