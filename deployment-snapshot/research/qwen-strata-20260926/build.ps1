[CmdletBinding()]
param([ValidateRange(1,8)][int]$Jobs=4)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$source=Join-Path $PSScriptRoot 'source'
$build=Join-Path $PSScriptRoot 'build-native'
$cuda='E:\Project_ANTIREZ-tools\cuda-13.3.1\toolkit'
$dev='C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'
$ninja='C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
if ((& git -C $source rev-parse HEAD).Trim() -ne '6da1f667e86558b152ab128edf3ebf77a80a9e57') { throw 'Unexpected Strata revision' }
$dep=Join-Path $source 'third_party\llama.cpp'
if ((& git -C $dep rev-parse HEAD).Trim() -ne '3cf03257f219afbe7334045ff7c6a06ac68c627d') { throw 'Unexpected ggml revision' }
$lines=& $env:ComSpec /d /s /c "`"$dev`" -arch=x64 -host_arch=x64 >nul && set"
if ($LASTEXITCODE) { throw 'Visual Studio environment failed' }
$devPath=$null
foreach($line in $lines) {
    $i=$line.IndexOf('='); if($i -le 0) {continue}
    $name=$line.Substring(0,$i); $value=$line.Substring($i+1)
    if($name.Equals('PATH',[StringComparison]::OrdinalIgnoreCase)) {
        if($value -match '(?i)\\VC\\Tools\\MSVC\\[^;]+\\bin\\HostX64\\x64') {$devPath=$value}
        elseif(-not $devPath) {$devPath=$value}
    } else {Set-Item -Path "Env:$name" -Value $value}
}
if(-not $devPath) {throw 'Compiler PATH missing'}
$env:PATH="$(Join-Path $cuda 'bin');$devPath"
$env:CUDA_PATH=$cuda
& cmake -S $source -B $build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" '-DCMAKE_CXX_COMPILER=C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe' '-DCMAKE_C_COMPILER=C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe' '-DCMAKE_BUILD_TYPE=Release' '-DSTRATA_ENABLE_CUDA=ON' '-DSTRATA_BUILD_TESTS=OFF' '-DSTRATA_PORTABLE=ON' '-DCMAKE_CUDA_ARCHITECTURES=120' "-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe" "-DCUDAToolkit_ROOT=$cuda" "-DSTRATA_GGML_DIR=$dep"
if($LASTEXITCODE) {throw 'CMake configure failed'}
& cmake --build $build --parallel $Jobs
if($LASTEXITCODE) {throw 'Build failed'}
