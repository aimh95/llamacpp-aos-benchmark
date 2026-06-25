# llamacpp-android-cpu-baseline

llama.cpp를 Android 기기에서 **CPU 전용**으로 빌드/실행하여 baseline 성능(pp/tg t/s)을 측정하기 위한 벤치마크 저장소입니다.

## 사전 준비

- Android NDK (`ANDROID_NDK_HOME` 환경변수 설정)
- CMake, Ninja
- adb (디버깅 모드로 연결된 Android 기기)
- git

## 디렉터리 구조

```
.
├── third_party/
│   └── llama.cpp/        # git submodule (직접 추가: git submodule add <url> third_party/llama.cpp)
├── models/
│   └── model.gguf        # 벤치마크에 사용할 gguf 모델 (symlink 또는 복사본, 직접 배치)
├── scripts/
│   ├── 00_env_check.sh       # 빌드/실행 환경 점검
│   ├── 01_build_android_cpu.sh   # NDK 툴체인으로 CPU 전용 빌드
│   ├── 02_download_model.sh  # 모델 다운로드 helper (옵션)
│   ├── 03_adb_push.sh         # 바이너리 + 모델을 기기에 push
│   ├── 04_run_cli.sh          # llama-cli sanity check
│   └── 05_run_bench.sh        # llama-bench 실행 및 로그 저장
├── logs/                  # 벤치마크 raw 로그
└── docs/
    ├── device_info.md     # 측정 기기 사양 기록
    └── cpu_baseline_result.md  # 측정 결과 기록
```

## Quickstart

```bash
git submodule add https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
cp /path/to/your-model.gguf models/model.gguf

./scripts/00_env_check.sh
./scripts/01_build_android_cpu.sh
./scripts/03_adb_push.sh
./scripts/04_run_cli.sh
./scripts/05_run_bench.sh
```
