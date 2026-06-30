# build_snapdragon

llama.cpp의 **Snapdragon(Hexagon HTP + Adreno OpenCL) Android 패키지**를 빌드/설치하는 도구.

대상: Samsung SM-F956N (SoC SM8650 / QTI, board_platform `pineapple`, ABI `arm64-v8a`), HTP candidate `V75`.

## 왜 필요한가: CPU-only 빌드와의 차이

지금까지 단말에 올려둔 `/data/local/tmp/llamacpp_cpu` 는 [scripts/01_build_android_cpu.sh](../../scripts/01_build_android_cpu.sh)로
NDK toolchain만 사용해 빌드한 **CPU-only** 패키지다. `libggml-cpu.so`만 있고 `libggml-hexagon.so` /
`libggml-htp-v75.so` / `libggml-opencl.so`가 없어서, `--list-devices`에 `HTP0`가 보이지 않는다.
즉 NPU 실행이 "실패"한 게 아니라 **HTP backend를 포함한 빌드 자체가 없었던 것**이다.

이 도구가 만드는 Snapdragon 패키지는 Hexagon SDK / OpenCL SDK까지 포함한 toolchain으로 빌드되어
`libggml-hexagon.so`(HTP 백엔드 로더)와 HTP 세대별 NPU 펌웨어(`libggml-htp-v73/75/79/81.so`),
`libggml-opencl.so`(Adreno GPU 백엔드)를 함께 만든다.

## 핵심 개념

| 용어 | 의미 |
| --- | --- |
| `GGML_HEXAGON` | cmake 옵션. `ON`이면 Hexagon NPU 백엔드(`ggml-hexagon`)를 빌드에 포함시킨다. |
| `GGML_OPENCL` | cmake 옵션. `ON`이면 Adreno GPU용 OpenCL 백엔드를 빌드에 포함시킨다. |
| `HTP0`~`HTP4` | llama.cpp가 노출하는 Hexagon NPU 디바이스 이름. `--device`/`-dev`로 선택한다 (`-ngl`은 GPU와 동일하게 동작). |
| `libggml-hexagon.so` | Hexagon 백엔드 로더(host aarch64에서 동작). 이게 없으면 HTP0가 절대 안 보인다. |
| `libggml-htp-vNN.so` | HTP 세대별(V73/V75/V79/V81) DSP(cdsp)에서 실제로 도는 NPU 펌웨어. 단말의 HTP candidate(여기서는 V75)와 맞는 버전이 필요하다. |
| wrapper script | `scripts/snapdragon/adb/run-*.sh` (host에서 실행, 내부적으로 `adb shell`을 호출). `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` 등 필요한 환경변수를 자동으로 잡아준다. 직접 `adb shell ./bin/llama-cli ...`를 칠 때 가장 흔한 실패 원인이 바로 이 두 환경변수를 빠뜨리는 것이다. |

자세한 공식 절차는 `third_party/llama.cpp/docs/backend/snapdragon/README.md` 참고.

## 사용 순서

### 0. 사전 준비: 공식 Snapdragon toolchain Docker

빌드는 Android NDK + Hexagon SDK + OpenCL SDK가 모두 설치된 공식 Docker 이미지 안에서 하는 것을 권장한다.

```bash
docker run -it -u $(id -u):$(id -g) --volume <llama.cpp-checkout>:/workspace \
  --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.7
[d]/> cd /workspace
```

### 1. 빌드

컨테이너 안, llama.cpp 소스 루트에서:

```bash
/workspace$ /path/to/llamacpp-aos-benchmark/tools/build_snapdragon/build_snapdragon_llamacpp.sh
```

`--src-dir`를 안 주면 현재 디렉터리를 llama.cpp 루트로 간주하고, `CMakeLists.txt`가 없으면
`third_party/llama.cpp` 서브모듈로 자동 fallback한다.

내부적으로 공식 4단계를 그대로 수행한다:

```bash
cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --preset arm64-android-snapdragon-release -B build-snapdragon
cmake --build build-snapdragon
cmake --install build-snapdragon --prefix pkg-snapdragon/llama.cpp
```

configure 로그에서 `GGML_HEXAGON=ON`, `GGML_OPENCL=ON`, `Including Hexagon backend`,
`Including OpenCL backend`를 찾아 `configure_markers.txt`에 FOUND/NOT_FOUND로 남긴다(참고용 경고일 뿐,
빌드 자체를 막지는 않는다 — cmake 출력 형식이 버전마다 살짝 다를 수 있기 때문).

추가로, `cmake --install`은 `scripts/snapdragon`을 패키지에 설치하지 않으므로, 이후 단계(check/install
스크립트의 wrapper 사용)를 위해 `scripts/snapdragon`을 `pkg-snapdragon/llama.cpp/scripts/snapdragon`으로
별도 복사한다 (실패해도 비치명적: 경고만 남고 빌드는 완료된 것으로 처리).

로그: `artifacts/build_snapdragon/build_<timestamp>/{configure,build,install}.log`, `pkg_path.txt`.

configure/build/install 중 하나라도 실패하면 **그 자리에서 즉시 중단**한다 (probe 도구들과 달리, 이전 단계가
실패하면 다음 단계를 계속하는 게 의미가 없기 때문). raw 로그는 삭제하지 않는다.

### 2. 패키지 점검

```bash
tools/build_snapdragon/check_snapdragon_pkg.sh --pkg-dir <build 단계가 출력한 pkg 경로>
```

필수 항목(`bin/llama-cli`, `bin/llama-bench`, `lib/libggml.so`, `lib/libggml-cpu.so`,
`lib/libggml-opencl.so`, `lib/libggml-hexagon.so`, `lib/libggml-htp-v75.so`) 중 하나라도 없으면
`summary.md`에서 `Overall: FAIL`로 표시되고 exit code 1을 반환한다. 권장 항목(다른 HTP 세대, adb wrapper
스크립트들)은 없어도 FAIL로 잡지 않고 목록만 보여준다.

### 3. 단말 설치

```bash
tools/build_snapdragon/install_snapdragon_pkg.sh \
  --serial <adb serial, 선택> \
  --pkg-dir <pkg 경로> \
  --model /path/to/model.gguf   # 선택
```

- 패키지는 `/data/local/tmp/llama.cpp` 로, 모델은 `/data/local/tmp/gguf/` 로 push한다.
- **기존 `/data/local/tmp/llamacpp_cpu`는 건드리지 않는다.** 새 패키지는 완전히 분리된 경로에 설치된다.
- 설치 후 `bin/llama-cli`, `lib/libggml-hexagon.so`, `lib/libggml-htp-v75.so`, `lib/libggml-opencl.so` 4개
  파일의 존재를 단말에서 직접 확인하고 `install_verify.txt`에 기록한다.
- `--list-devices`를 wrapper(`scripts/snapdragon/adb/run-tool.sh llama-cli --list-devices`, 패키지에
  포함된 경우)와 직접 실행(`LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` 직접 지정) 둘 다 시도해서 각각 로그로 남긴다.
- 무엇이 실패하든 끝까지 진행하고 로그를 남긴다. exit code는 4개 필수 파일 검증 결과를 반영한다.

로그: `artifacts/build_snapdragon/install_<timestamp>/`.

### 4. 실제 동작 확인

설치만으로는 "HTP0가 실제로 도는지"까지는 보장하지 않는다. 다음 단계로 [tools/htp_smoke](../htp_smoke/README.md)를
실행해서 실제 HTP0 실행 smoke test를 돌려라 (`--remote-dir /data/local/tmp/llama.cpp`).

## HTP0가 안 보일 때 체크리스트

1. `check_snapdragon_pkg.sh` 결과가 PASS인가? (`libggml-hexagon.so` / `libggml-htp-v75.so` /
   `libggml-opencl.so`가 로컬 패키지에 실제로 있는가)
2. `install_verify.txt`에서 같은 4개 파일이 단말에도 **실제로 push됐는지** 확인했는가? (push 실패는
   조용히 넘어갈 수 있다 — `push_pkg.log`를 봐라)
3. `--pkg-dir`의 basename이 `llama.cpp`인가? (`adb push <dir> /data/local/tmp/`는 dir의 basename으로
   nesting되므로, 이름이 다르면 `/data/local/tmp/llama.cpp`가 아닌 다른 경로에 깔린다)
4. 직접 실행 시 `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH`를 `lib/`로 지정했는가? (안 그러면 `libggml-hexagon.so`를
   못 찾아서 백엔드 자체가 로드되지 않고 CPU만 보일 수 있다)
5. 단말의 HTP candidate(V75)와 설치된 `libggml-htp-vNN.so` 버전이 맞는가? (V75 candidate는 파일명 기반
   추정치다 — [tools/device_probe](../device_probe/README.md) 참고)
6. `list_devices_wrapper.log` / `list_devices_direct.log`에 `HTP0`가 실제로 나열되는가?

## 성공 판정 기준

**HTP V75는 단말 파일명 기반 candidate일 뿐, 확정값이 아니다.** 실제 NPU 성공 판정에는 다음이 모두
필요하다 (자동 판정은 [tools/htp_smoke](../htp_smoke/README.md)가 candidate 수준으로 시도한다):

- `--list-devices`에 `HTP0` visible
- runtime 로그에 `ggml-hex: Hexagon backend` / `ggml-hex: Hexagon Arch version vNN` 같은 초기화 로그
- `load_tensors: offloaded N/N layers` 등 실제 offload 로그
- `fallback`/`cpu fallback`/`device not found` 같은 실패 키워드가 없음

이 도구(`build_snapdragon`)는 빌드와 설치까지만 책임진다. 위 항목의 최종 확인은 `tools/htp_smoke`로 한다.
