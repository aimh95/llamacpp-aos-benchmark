# EXAONE4 ixi_GEN 1.2B — llama.cpp Snapdragon HTP(NPU) 추론 가이드

`models/260526_ixi_GEN_1.2B_vocab_trim` (vocab-trimmed EXAONE 4.0 1.2B) 모델을
Snapdragon Hexagon NPU(HTP0)에서 llama.cpp로 추론하기 위한 절차서.

호스트(x86 Linux)에서 GGUF 변환·양자화·HTP 패키지 빌드를 하고, adb로 단말에 설치해
HTP0 백엔드로 실행한다.

---

## 진행 상태 (2026-07-13 기준)

| 단계 | 상태 | 산출물 / 비고 |
| --- | --- | --- |
| Step 1 — HF → f16 GGUF | ✅ 완료·검증 | `models/ixi_gen_1p2b_f16.gguf` (2.4G, n_tensors=333) |
| Step 2 — Q4_0 양자화 | ✅ 완료·검증 | `models/ixi_gen_1p2b_q4_0.gguf` (**679 MiB, 4.73 BPW**) |
| 호스트 CPU 로드·생성 확인 | ✅ | 한국어/숫자 토큰화 정상 (`llama-completion`으로 확인) |
| 토크나이저 해시 등록 | ✅ 적용됨 | `conversion/base.py`에 `2ae277ad…` → `exaone` 매핑 추가 (아직 커밋 안 함) |
| Step 3 — HTP 패키지 빌드 | ⬜ 미실행 | Snapdragon toolchain Docker 필요 |
| Step 4 — 단말 설치 | ⬜ 미실행 | `install_snapdragon_pkg.sh --model models/ixi_gen_1p2b_q4_0.gguf` |
| Step 5 — HTP0 실행/검증 | ⬜ 미실행 | 단말(SM-F956N) 필요 |

> **롱컨텍스트 주의:** Step 1 변환 로그에서 `rope scaling type = NONE`으로 기록됐다(HF config에는
> `rope_scaling: llama3, factor 16.0`이 있음). 짧은 컨텍스트 추론엔 무관하지만, 32K 롱컨텍스트를 쓸
> 계획이면 RoPE 스케일링이 GGUF에 반영됐는지 별도 확인이 필요하다.

---

## 0. 대상 / 전제

| 항목 | 값 |
| --- | --- |
| 모델 | `models/260526_ixi_GEN_1.2B_vocab_trim/model.safetensors` (2.4GB, bf16) |
| 아키텍처 | `Exaone4ForCausalLM` / `model_type: exaone4` |
| 구조 | hidden 2048, layers 30, heads 32 (KV 8), intermediate 4096, head_dim 64 |
| vocab | 65536 (GPT2 BPE, `vocab.json`+`merges.txt`) |
| 특이점 | **`tie_word_embeddings: true`** — 입력 임베딩과 출력 projection이 같은 텐서(`token_embd`) 공유 |
| smoothed 변형 | `smoothed_a0.5/model_smoothed.safetensors` — SmoothQuant(α=0.5) 전처리본(저비트 양자화 정확도용) |
| 단말 | Samsung SM-F956N (SoC SM8650, board `pineapple`, ABI `arm64-v8a`) |
| HTP candidate | **V75** (`libggml-htp-v75.so`) — 파일명 기반 추정, [tools/device_probe](../tools/device_probe/README.md) 참고 |
| llama.cpp | `third_party/llama.cpp` @ `snapdragon-htp-patch` (EXAONE4 변환·런타임 지원, HTP 패치 포함) |

---

## 1. 먼저 읽을 핵심 제약

### (1) HTP는 Q4_0 계열 weight일 때만 실제로 오프로드된다
Hexagon 백엔드의 matmul/REPACK 지원 타입은
`ggml-hexagon.cpp`의 `ggml_hexagon_is_repack_type()` 기준으로 다음뿐이다:

> **Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4** (+ F16/F32 passthrough)

실전·공식 예제 경로는 전부 **Q4_0**이다 (`HTP0-REPACK` 버퍼, `test-backend-ops MUL_MAT type_a=q4_0`).
- 현재 저장소에 있는 `models/exaone4_1p2b_q8_0_*.gguf`는 **Q8_0**이라 CPU엔 돌지만 HTP 이득이 제한적이고, 이 모델용도 아니다(공개 EXAONE 기반).
- **NPU 추론을 하려면 이 모델을 Q4_0 GGUF로 새로 만들어야 한다.** (아래 Step 2)

### (2) tie_word_embeddings=true → get_rows(임베딩 조회)가 관전 포인트
이 모델은 임베딩 텐서 하나를 입력 조회(`GET_ROWS`)와 출력 projection(`MUL_MAT`) 양쪽에 쓴다.
기본 llama.cpp HTP 백엔드는 `GET_ROWS`의 src0가 F32일 때만 처리하고 Q4_0면 CPU로 fallback한다.
이 fork에는 그걸 HTP로 강제 라우팅하는 **`FORCE_GET_ROWS_HTP`** 패치가 들어가 있다
(`ggml-hexagon.cpp:86`, 컴파일타임 매크로 **기본 ON**). 즉 별도 env 없이 빌드에 이미 반영됨.
- 실행 시 `ggml-hex: [FORCE_GET_ROWS_HTP] ...` 로그가 나오는지 확인하면 이 경로가 실제로 탔는지 알 수 있다.
- tied 임베딩의 GGUF 표현(`output.weight` 유무)과 HTP 거동 비교는
  [experiments/tied_embedding_exaone4_1p2b](../experiments/tied_embedding_exaone4_1p2b/summary.md) 참고.

### (3) CPU-only 빌드로는 HTP0가 안 보인다
기존 `build-android`(NDK CPU-only)에는 `libggml-hexagon.so`가 없어 `--list-devices`에 HTP0가 안 뜬다.
반드시 [tools/build_snapdragon](../tools/build_snapdragon/README.md)의 Snapdragon 패키지를 써야 한다.

---

## 2. 전체 파이프라인

```
[HOST x86]                                              [DEVICE arm64 / SM8650]
safetensors ──convert_hf_to_gguf──▶ f16 GGUF
     │                                   │
     │                            llama-quantize
     │                                   ▼
     │                              Q4_0 GGUF ───adb push──▶ /data/local/tmp/gguf/
     │
docker(build_snapdragon) ─▶ pkg-snapdragon/llama.cpp ─adb push─▶ /data/local/tmp/llama.cpp/
                                                                        │
                                                            run-completion.sh / htp_smoke  (D=HTP0)
```

1. **Step 1** HF safetensors → GGUF(f16)   — host, `convert_hf_to_gguf.py`
2. **Step 2** GGUF(f16) → **Q4_0**          — host, `llama-quantize` (host native 빌드 필요)
3. **Step 3** Snapdragon HTP 패키지 빌드     — docker, `build_snapdragon_llamacpp.sh`
4. **Step 4** 패키지 + Q4_0 GGUF 단말 설치   — `install_snapdragon_pkg.sh`
5. **Step 5** HTP0 실행 및 검증              — `run_htp0_smoke.sh` / wrapper 스크립트

모든 명령은 저장소 루트(`llamacpp-aos-benchmark/`)에서 실행한다고 가정한다.

---

## Step 1 — HF safetensors → GGUF (f16)

> **주의 1 (Python 환경):** 변환에 필요한 `torch`/`safetensors`는 리포의 **`.venv`에만** 설치돼 있다.
> 현재 셸의 `python3`(시스템 파이썬)엔 없으니 반드시 `.venv/bin/python`으로 실행한다.

> **주의 2 (토크나이저 해시 등록 — 이 모델의 필수 선행 패치):** 이 vocab-trim 모델은 pre-tokenizer 해시가
> `convert_hf_to_gguf.py`에 미등록이라, 그냥 돌리면
> `NotImplementedError: BPE pre-tokenizer was not recognized`로 죽는다. 이미 아래 매핑을
> `third_party/llama.cpp/conversion/base.py`(`get_vocab_base_pre`)에 추가해 두었다:
> ```python
> if chkhsh == "2ae277ad2aa9ee6132c39e23b54119011b2053449a675d9e2cdf6b66dc09f6f6":
>     res = "exaone"   # Digits(individual)+ByteLevel == LLAMA_VOCAB_PRE_TYPE_EXAONE (GPT2 아님!)
> ```
> **중요:** 공개 EXAONE-4.0은 `exaone4`(=GPT2, 숫자를 `\p{N}+`로 묶음)를 쓰지만, 이 트림 모델의
> tokenizer는 `Digits(individual_digits=true)`가 추가돼 숫자를 한 자리씩 분리한다. 그래서 pre-type은
> `exaone4`가 아니라 **`exaone`**(개별 숫자 `\p{N}` + GPT2)이 맞다. `exaone4`로 매핑하면 숫자 토큰화가
> 어긋난다. 서브모듈을 reset/재클론하면 이 매핑이 사라지므로 다시 넣어야 한다.

```bash
.venv/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
  models/260526_ixi_GEN_1.2B_vocab_trim \
  --outfile models/ixi_gen_1p2b_f16.gguf \
  --outtype f16
```

- `convert_hf_to_gguf.py`는 `--outtype`으로 `f32/f16/bf16/q8_0/...`만 지원하고 **q4_0는 없다** → Q4_0는 Step 2에서 만든다.
- 로그 끝에 `Model successfully exported ...` 가 나오면 성공. `token_embd` 등 텐서가 정상 매핑되는지,
  tokenizer(BPE) 관련 경고가 없는지 확인.
- (검증됨) 이 저장소에서 실제로 실행해 `models/ixi_gen_1p2b_f16.gguf`(2.4G, n_tensors=333)까지 생성 완료.
- (선택) **smoothed 변형으로 변환**: 정확도 우선이면 SmoothQuant 본을 쓴다. `smoothed_a0.5/`에는
  safetensors만 있으므로 config/tokenizer를 갖춘 폴더를 따로 구성해야 한다:
  ```bash
  mkdir -p models/ixi_gen_smoothed_hf
  cd models/ixi_gen_smoothed_hf
  ln -sf ../260526_ixi_GEN_1.2B_vocab_trim/*.json .
  ln -sf ../260526_ixi_GEN_1.2B_vocab_trim/merges.txt .
  ln -sf ../260526_ixi_GEN_1.2B_vocab_trim/*.jinja .
  ln -sf ../260526_ixi_GEN_1.2B_vocab_trim/smoothed_a0.5/model_smoothed.safetensors ./model.safetensors
  cd ../..
  python3 third_party/llama.cpp/convert_hf_to_gguf.py models/ixi_gen_smoothed_hf \
    --outfile models/ixi_gen_1p2b_smoothed_f16.gguf --outtype f16
  ```

---

## Step 2 — Q4_0 양자화 (host native `llama-quantize`)

현재 저장소의 `build-android/bin/llama-quantize`는 **arm64 바이너리**라 호스트에서 못 돈다.
호스트용 CPU 빌드를 한 번 만들어 두면 된다(HTP/OpenCL 불필요, 수분 소요):

```bash
cmake -S third_party/llama.cpp -B build-host \
  -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --target llama-quantize -j"$(nproc)"
```

양자화:

```bash
build-host/bin/llama-quantize \
  models/ixi_gen_1p2b_f16.gguf \
  models/ixi_gen_1p2b_q4_0.gguf \
  Q4_0
```

- 결과 `models/ixi_gen_1p2b_q4_0.gguf` (검증됨: **679 MiB, 4.73 BPW**)가 단말에 올릴 최종 모델이다.
- (검증됨) 호스트 CPU에서 로드·생성 확인 완료 — 한국어/숫자 토큰화 정상.
  단, **이 fork의 `llama-cli`는 `-no-cnv`를 지원하지 않는다**(→ 대화모드로 빠져 stdin EOF 루프).
  비대화형 확인은 `llama-completion`을 써라:
  ```bash
  build-host/bin/llama-completion -m models/ixi_gen_1p2b_q4_0.gguf -p "대한민국의 수도는" -n 32 --temp 0
  ```
- (선택, 정확도) `--imatrix <imatrix.dat>`로 importance matrix를 주면 Q4_0 손실을 줄일 수 있다.
  imatrix는 `build-host/bin/llama-imatrix`로 캘리브레이션 텍스트에서 생성.
- smoothed 본을 만들었다면 동일하게 `..._smoothed_f16.gguf → ..._smoothed_q4_0.gguf`.

---

## Step 3 — Snapdragon HTP 패키지 빌드 (Docker)

상세는 [tools/build_snapdragon/README.md](../tools/build_snapdragon/README.md). 요약:

> **주의 (마운트 경로 — 자주 막히는 지점):** `third_party/llama.cpp`만 `/workspace`로 마운트하면
> 안 된다. 빌드 스크립트는 `tools/build_snapdragon/`(스크립트 위치)와 `artifacts/`(로그 출력),
> `third_party/llama.cpp`(실제 빌드)를 **모두** 필요로 한다. submodule만 마운트하면 컨테이너 안에
> 스크립트가 없어 `No such file or directory`로 죽는다. **저장소 루트 전체를 마운트**해야 한다.

```bash
# 호스트에서: 저장소 루트에서 실행. 저장소 전체를 /workspace로 마운트한다.
cd /home/iptv-infra/workspace/llamacpp-aos-benchmark          # 저장소 루트
docker run -it -u $(id -u):$(id -g) \
  --volume "$(pwd):/workspace" \
  --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.7

# 컨테이너 안 (/workspace = 저장소 루트)
cd /workspace
tools/build_snapdragon/build_snapdragon_llamacpp.sh
```

- 스크립트가 경로를 자동 판정한다: `/workspace`엔 `CMakeLists.txt`가 없으므로 SRC_DIR을
  `third_party/llama.cpp`로 fallback하고, 빌드 로그는 `artifacts/build_snapdragon/build_<ts>/`,
  패키지는 `third_party/llama.cpp/pkg-snapdragon/llama.cpp`에 만든다.
- 내부적으로 `arm64-android-snapdragon-release` preset으로 `GGML_HEXAGON=ON`, `GGML_OPENCL=ON` 빌드 후
  `pkg-snapdragon/llama.cpp`로 install 한다.
- 산출 `lib/`에 `libggml-hexagon.so` + `libggml-htp-v75.so`(단말 세대와 일치) + `libggml-opencl.so`가 있어야 한다.

빌드 후 패키지 점검(호스트):

```bash
tools/build_snapdragon/check_snapdragon_pkg.sh --pkg-dir <build 로그가 출력한 pkg 경로>
# summary.md 의 Overall: PASS 확인
```

---

## Step 4 — 단말 설치

```bash
tools/build_snapdragon/install_snapdragon_pkg.sh \
  --serial <adb serial> \
  --pkg-dir <Step 3의 pkg 경로 (basename이 반드시 llama.cpp)> \
  --model  models/ixi_gen_1p2b_q4_0.gguf
```

- 패키지는 `/data/local/tmp/llama.cpp/`, 모델은 `/data/local/tmp/gguf/`로 push된다.
- 기존 `/data/local/tmp/llamacpp_cpu`(CPU baseline)는 건드리지 않는다.
- `install_verify.txt`에서 `libggml-hexagon.so` / `libggml-htp-v75.so` / `libggml-opencl.so`가
  단말에 실제로 올라갔는지 확인.

---

## Step 5 — HTP0 실행 & 검증

### 5-1. Smoke test (권장 시작점)

```bash
tools/htp_smoke/run_htp0_smoke.sh \
  --serial <adb serial> \
  --remote-dir /data/local/tmp/llama.cpp \
  --model /data/local/tmp/gguf/ixi_gen_1p2b_q4_0.gguf \
  --backend-device HTP0 \
  --prompt "온디바이스 AI를 한 문장으로 설명해줘." \
  --n-predict 64 --ctx-size 512 --ngl 99
```

결과는 `artifacts/htp_smoke/<timestamp>/summary.md`. raw는 `htp0_smoke.log`.

### 5-2. wrapper로 직접 실행 (패키지 내 스크립트)

단말/컨테이너가 아닌 host에서 adb 경유로 도는 wrapper:

```bash
# 예시 (scripts/snapdragon/adb/*.sh — LD_LIBRARY_PATH/ADSP_LIBRARY_PATH 자동 설정)
M=ixi_gen_1p2b_q4_0.gguf D=HTP0 \
  third_party/llama.cpp/scripts/snapdragon/adb/run-completion.sh \
  -p "what is on-device AI?"

# 벤치마크
M=ixi_gen_1p2b_q4_0.gguf D=HTP0 \
  third_party/llama.cpp/scripts/snapdragon/adb/run-bench.sh -p 128 -n 64
```

> 직접 `adb shell ./llama-cli ...`를 칠 거면 **`LD_LIBRARY_PATH`와 `ADSP_LIBRARY_PATH`를 `lib/`로**
> 반드시 지정해야 한다(누락이 HTP0 안 보이는 가장 흔한 원인).

### 5-3. 성공 판정 체크리스트

`htp0_smoke.log` / 실행 로그에서:

- [ ] `--list-devices`에 **HTP0** 표시
- [ ] `ggml-hex: Hexagon backend ...` / `ggml-hex: Hexagon Arch version vNN` 초기화 로그
- [ ] `load_tensors: offloaded 31/31 layers to GPU` (30 blocks + output layer)
- [ ] `HTP0-REPACK model buffer size = ... MiB` (> 0)
- [ ] **`ggml-hex: [FORCE_GET_ROWS_HTP] ...`** 로그 (tied 임베딩 조회가 HTP로 갔다는 신호)
- [ ] `fallback` / `cpu fallback` / `device not found` / `unsupported op` 키워드 **없음**
- [ ] `llama_perf_context_print` 성능 수치 정상 출력

하나라도 실패하면 [build_snapdragon README의 "HTP0가 안 보일 때 체크리스트"](../tools/build_snapdragon/README.md) 참고.

---

## 6. 디버깅용 env (실행 시)

`ggml-hexagon.cpp` 기준, wrapper 커맨드 앞이나 `adb shell` env로 지정:

| 변수 | 용도 |
| --- | --- |
| `GGML_HEXAGON_VERBOSE=1` | op별 HTP 배치 로그(어떤 matmul/get_rows가 HTP인지 CPU인지) |
| `GGML_HEXAGON_PROFILE=1` | op별 usec/cycle 프로파일 |
| `GGML_HEXAGON_NDEV=1` | 세션(디바이스) 수. 1.2B는 1개면 충분 |
| `GGML_HEXAGON_OPFILTER="FLASH_ATTN_EXT"` | 특정 op를 HTP에서 제외(CPU/GPU fallback) — 문제 op 격리용 |
| `GGML_HEXAGON_HOSTBUF=1` | REPACK 버퍼 관련 op 테스트 시 |

`GGML_HEXAGON_VERBOSE=1`은 이 모델의 tied 임베딩 거동을 확인할 때 특히 유용하다
(`token_embd` / `get_rows` 노드가 `HTP0`인지 확인).

---

## 7. 정확도 / 트러블슈팅 메모

- **Q4_0 품질 저하가 크면**: (1) smoothed_a0.5 본으로 변환(Step 1 선택), (2) imatrix 적용(Step 2),
  (3) 임베딩/출력만 Q8_0로 두는 mixed 양자화(`llama-quantize --token-embedding-type q8_0 --output-tensor-type q8_0`) 검토.
- **HTP0는 뜨는데 offload 0 / CPU fallback**: weight가 Q4_0가 아닌지(Step 2 확인), HTP 세대(V75) 라이브러리 불일치인지 확인.
- **`unsupported op`**: `GGML_HEXAGON_OPFILTER`로 해당 op를 제외해 나머지라도 HTP로 돌려보고, op는 CPU fallback시켜 범위를 좁힌다.
- **tokenizer/BOS·EOS 이슈**: 이 모델은 `bos=1`, `eos=361`, `pad=0`, GPT2 BPE. 출력이 이상하면 `--chat-template`/특수토큰 매핑 우선 확인.

---

## 참고

- 공식 백엔드 문서: `third_party/llama.cpp/docs/backend/snapdragon/README.md`
- 빌드/설치: [tools/build_snapdragon/README.md](../tools/build_snapdragon/README.md)
- Smoke test: [tools/htp_smoke/README.md](../tools/htp_smoke/README.md)
- tied 임베딩 실험: [experiments/tied_embedding_exaone4_1p2b/summary.md](../experiments/tied_embedding_exaone4_1p2b/summary.md)
- HTP 패치: `patches/0001-htp-optrace-embtrace-force-get-rows.patch`
