# device_probe

Android 단말의 SoC / ABI / QNN-HTP 관련 라이브러리 존재 여부를 수집해 Markdown/CSV 리포트를 만드는 도구.

llama.cpp 기반 Android 온디바이스 LLM PoC의 "현재 단말 및 SoC / HTP 세대 확인" 단계에서 사용한다.
**목적은 NPU 실행 성공 판정이 아니라, QNN/HTP feasibility 검토를 위한 단말 정보표 생성이다.**

## 사용법

```bash
# 단말이 하나만 연결된 경우
./tools/device_probe/collect_android_device_info.sh

# 여러 대가 연결된 경우 대상 지정
DEVICE_SERIAL=<adb serial> ./tools/device_probe/collect_android_device_info.sh
```

요구사항: `adb`, `python3`(표준 라이브러리만 사용) 가 PATH에 있어야 하고, USB 디버깅이 허용된 기기가 연결되어 있어야 한다. root 권한은 필요 없다.

## 출력

`artifacts/device_probe/<timestamp>/` 아래에 다음 파일이 생성된다.

| 파일 | 내용 |
| --- | --- |
| `devices.txt` | `adb devices -l` 결과 |
| `getprop.txt` | `getprop` 전체 |
| `cpuinfo.txt` | `/proc/cpuinfo` |
| `meminfo.txt` | `/proc/meminfo` |
| `uname.txt` | `uname -a` |
| `pm_features.txt` | `pm list features` |
| `host_env.txt` | host의 `ANDROID_HOME` / `ANDROID_NDK_HOME` / `ANDROID_NDK` / `QNN_SDK_ROOT` / `QAIRT_SDK_ROOT` |
| `qnn_libs.txt` | `/vendor`, `/system`, `/product`, `/odm`, `/apex`, `/data/local/tmp` 에서 찾은 QNN/HTP/OpenCL/RPC 라이브러리 경로 (permission denied는 무시) |
| `device_info.md` | 위 raw 데이터를 정리한 Markdown 리포트 |
| `device_info.csv` | 같은 내용의 CSV |

raw 파일은 삭제/가공 없이 그대로 보존되고, `device_info.md`/`device_info.csv`만 그로부터 파생된다.

## device_info 필드

`manufacturer`, `device_model`, `device_name`, `android_release`, `android_sdk`, `build_fingerprint`,
`abi_primary`, `abi_list`, `board_platform`, `hardware`, `soc_model`, `soc_manufacturer`,
`htp_candidate_versions_from_lib_name`, `qnn_htp_skel_count`, `qnn_htp_stub_count`, `qnn_core_lib_count`,
`cdsprpc_found`, `opencl_found`.

값을 못 찾으면 `NOT_FOUND`로 기록된다.

## HTP 세대 추정 (candidate only)

`libQnnHtpV75Skel.so`, `libQnnHtpV75Stub.so` 같은 파일명에서 `V75`를 candidate로 추출한다.
**이 값은 파일명 기반 추정치이며 확정값이 아니다.** 실제 NPU(HTP) 실행 가능 여부는 runtime에서
QNN/HTP backend initialization 로그를 직접 확인해야 한다. `device_info.md`에도 동일한 주의사항이 명시된다.

## 기존 raw 데이터 재파싱

raw 파일만 있고 리포트를 다시 만들고 싶다면:

```bash
python3 tools/device_probe/parse_device_info.py artifacts/device_probe/<timestamp>
```
