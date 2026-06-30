# htp_smoke

llama.cpp의 Snapdragon/Hexagon **HTP0 backend smoke test**를 자동화하는 도구.

대상: Samsung SM-F956N (SoC SM8650 / QTI, board_platform `pineapple`, ABI `arm64-v8a`), HTP candidate `V75`
([device_probe](../device_probe/README.md) 결과 기준).

**목적은 벤치마크가 아니다.** HTP0 backend가 보이고, 초기화되고, 모델이 offload되는지를 빠르게 확인하기 위한
smoke test이며, CPU/GPU 성능 측정은 다루지 않는다 (CPU baseline은 별도 도구에서 이미 확인됨).

## 사용법

```bash
./tools/htp_smoke/run_htp0_smoke.sh

# 옵션 지정 (모두 기본값 있음)
./tools/htp_smoke/run_htp0_smoke.sh \
  --serial <adb serial> \
  --remote-dir /data/local/tmp/llama.cpp \
  --model models/llama-3.2-1b-instruct-q4_0.gguf \
  --backend-device HTP0 \
  --prompt "Explain on-device AI in three short sentences." \
  --n-predict 64 \
  --ctx-size 512 \
  --ngl 99
```

`--serial` 대신 `DEVICE_SERIAL` 환경변수로도 지정 가능. 여러 단말이 연결된 경우 반드시 지정해야 한다.

요구사항: `adb`, `python3`(표준 라이브러리만 사용). root 권한 불필요.

단말의 `$REMOTE_DIR`에 `scripts/llama-cli.sh` wrapper가 있으면 그것을 우선 사용하고,
없으면 `$REMOTE_DIR/llama-cli`를 직접 실행한다.

## 출력

`artifacts/htp_smoke/<timestamp>/` 아래에 생성된다.

| 파일 | 내용 |
| --- | --- |
| `run_params.txt` | 이번 실행에 사용된 옵션 값 + 실제 실행 커맨드 |
| `devices.txt` | `adb devices -l` 결과 |
| `wrapper_check.txt` | wrapper 존재 여부 체크 결과 (`yes`/`no`) |
| `list_devices.log` | `--list-devices` (혹은 wrapper의 동일 기능) 출력 |
| `qnn_libs_before_run.txt` | 실행 직전 단말의 QNN/HTP/OpenCL/RPC 라이브러리 재수집 결과 |
| `search_errors.txt` | 라이브러리 검색 중 발생한 stderr |
| `htp0_smoke.log` | HTP0 실행 stdout/stderr |
| `exit_code.txt` | HTP0 실행의 종료 코드 |
| `logcat_qnn_htp.txt` | 실행 후 `adb logcat -d`에서 `qnn\|htp\|hexagon\|cdsp\|adsprpc\|rpc\|ggml\|llama` 키워드로 필터링한 결과 |
| `logcat_errors.txt` | logcat 수집 중 stderr |
| `summary.md` / `summary.csv` | 위 raw 로그를 정리한 최종 리포트 |

raw 로그는 삭제되지 않고 그대로 보존된다. 이 스크립트의 최종 exit code는 HTP0 실행 자체의 exit code와 동일하다
(수집/리포트 단계 실패는 경고만 출력하고 계속 진행한다).

## summary 필드 (candidate based on logs)

`parse_htp_smoke_log.py`가 `htp0_smoke.log` + `logcat_qnn_htp.txt`(런타임 신호)와 `list_devices.log`(가시성)를
키워드 기반으로 판정한다. **모든 판정은 candidate이며, 최종 성공/실패 판정은 사람이 raw log를 보고 직접 내려야 한다.**

| 필드 | 의미 |
| --- | --- |
| `htp_device_visible` | `list_devices.log`에 backend device 이름(기본 `HTP0`)이 보이는지 |
| `htp_runtime_initialized` | Hexagon/QNN backend 초기화 관련 키워드 후보 |
| `qnn_library_load_detected` | QNN 라이브러리 로드 관련 키워드 후보 |
| `skel_stub_keyword_detected` | `skel`/`stub` 키워드 존재 여부 |
| `cdsp_rpc_keyword_detected` | `cdsp`/`adsprpc`/`rpc` 키워드 존재 여부 |
| `offload_detected` / `offloaded_layers_raw` | `offloaded N/M layers` 류 패턴 매치 여부와 원문 |
| `cpu_fallback_suspected` | fallback/offload 0 등 CPU로 떨어졌다고 의심되는 신호 |
| `failure_keywords` | 발견된 실패 키워드 목록 (`fallback`, `device not found`, `failed to load`, `unsupported op/tensor`, `segmentation fault` 등) |
| `llama_perf_detected` | `llama_perf_context_print` 류 성능 출력 존재 여부 |
| `prompt_eval_time_ms`, `eval_time_ms`, `total_time_ms`, `tok_per_sec` | 가능하면 추출한 성능 수치 (`tok_per_sec`는 eval/generation 구간 기준) |

값을 판단할 근거가 없으면 `UNKNOWN`/`NOT_FOUND`로 기록된다.

## 기존 raw 로그 재파싱

```bash
python3 tools/htp_smoke/parse_htp_smoke_log.py artifacts/htp_smoke/<timestamp>
```
