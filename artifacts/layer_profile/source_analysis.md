# source_analysis — GGML scheduler & Hexagon HTP backend (layer profiling 근거)

대상 checkout: `third_party/llama.cpp` @ `e6b88e58446b72e9c11334a3f7a52fb73fd3b4a2` (snapdragon-htp-patch).
목적: layer/op별 **assigned backend / executed backend / timing**을 계측할 정확한 hook 지점을 소스에서 확정.
모든 항목은 실측 `file:line` 기준 (추측 아님).

---

## 1. 세 가지 "backend" 레벨의 실제 위치

| 레벨 | 의미 | 소스 위치 |
| --- | --- | --- |
| **supports-op** (capability) | backend가 op를 지원 가능한가 | 스케줄러가 `ggml_backend_supports_op` 호출: `ggml/src/ggml-backend.cpp:1006` (`set_if_supported`), 실제 판정은 Hexagon `ggml_backend_hexagon_device_supports_op` (ggml-hexagon.cpp, 로그 `supports-op ... yes/no`). **capability일 뿐, 최종 배정 아님.** |
| **assigned backend** (scheduler 최종 배정) | graph node가 실제로 어느 backend에 배정됐나 | node별 `tensor_backend_id(node)` 매크로 `ggml/src/ggml-backend.cpp:831`; 조회 API `ggml_backend_sched_get_tensor_backend(sched, node)` (backend.cpp, print_assignments에서 사용 `:967`). 배정 로직 = `ggml_backend_sched_split_graph` `:1014`. |
| **executed backend** (실제 compute 호출) | compute 함수가 실제로 호출된 backend | `ggml_backend_sched_compute_splits` `:1553` 안에서 split마다 `ggml_backend_graph_compute_async(split_backend, &split->graph)` `:1722` (그리고 fused 경로 `:1744`). `split_backend = sched->backends[split->backend_id]`. |

→ **Excel의 CPU/HTP 판정은 assigned + executed 기준으로 작성해야 함.** supports-op(예: 사용자가 본 `token_embd ... f16 ... CPU ... no`)는 후보 정보일 뿐이다.

---

## 2. Scheduler graph split / assignment (배정 파이프라인)

`ggml_backend_sched_split_graph` `ggml/src/ggml-backend.cpp:1014`:
- pass별로 각 node에 backend_id 부여. 배정 원인은 `SET_CAUSE(node, ...)` `:870` 로 문자열 기록되고 `GET_CAUSE(node)` `:871` 로 조회 (예: `2.sup`=supports 기반, `1.wgt`=weight 위치, `1.inp`=입력, `1.dst`, `1.vsrc`).
- weight 소유 backend 우선 배정: `ggml_backend_sched_backend_from_buffer` `:845`.
- 인접 동일 backend node들을 **split(subgraph)** 으로 묶음. split 배열: `sched->splits[]`, 각 split은 `i_start`, `backend_id`, `n_inputs`, `inputs[]` 필드 보유 (print_assignments `:948-957` 에서 사용).

기존 디버그 덤프 `ggml_backend_sched_print_assignments` `:945` (env `GGML_SCHED_DEBUG>=2` `:1784`):
- `## SPLIT #<i>: <backend> # <n> inputs` `:950`
- node별: `node #<idx> (<op>): <name> [<assigned_backend> <cause>] ...` `:968` — **이미 assigned backend를 노드별로 찍음** (단, DEBUG 레벨).

이 repo 자체 계측 **OPTRACE** (`GGML_OP_TRACE=1`): `[OPTRACE][SCHED]` split별 backend/n_nodes/GET_ROWS `ggml/src/ggml-backend.cpp:1541~1600`.

---

## 3. 실제 실행 / copy / synchronize (executed + timing hook 지점)

`ggml_backend_sched_compute_splits` `ggml/src/ggml-backend.cpp:1553` — **timing 계측의 핵심 함수**:

| 동작 | 위치 | 계측 의미 |
| --- | --- | --- |
| split 입력 준비 시 backend 간 **tensor copy** | `:1611`, `:1715` (`ggml_backend_tensor_copy`) | CPU↔HTP copy time |
| copy 전후 **synchronize** | `:1609`, `:1617`, `:1632`, `:1651`, `:1709`, `:1713` | synchronize 대기 |
| split **compute (executed)** | `:1722` `ggml_backend_graph_compute_async(split_backend, &split->graph)`; fused `:1744` | split(≈block) compute time |
| split 후 **synchronize** | `:1750` (그리고 `:1528` 전체 sync) | dispatch→완료 대기 |

`ggml_backend_graph_compute(_async)` 정의 `:444/:450`, `ggml_backend_synchronize` `:414`, `ggml_backend_tensor_copy(_async)` `:477/:500`.

**타이밍 hook 결론:** split loop(`:1553`) 안에서 각 split의 (copy, compute_async, synchronize) 구간에 timestamp를 감싸면, **노드별 강제 sync 없이** split(대부분 block 경계) 단위 executed-backend 시간을 얻는다. split↔node 매핑은 `split->i_start` + graph nodes로 복원.

---

## 4. Hexagon HTP dispatch / RPC / synchronize (batch 단위 특성)

`ggml/src/ggml-hexagon/ggml-hexagon.cpp`:
- DSP 통신은 **dspqueue** (`#include <dspqueue.h>` `:34`, `dspqueue_t queue` `:208`).
- **dispatch (enqueue)**: `flush_batch()` `:1563` → `dspqueue_write(queue, ...)` `:1579`.
- **synchronize (완료 대기)**: `flush_pending()` `:1524` / `flush()` `:1593` → `dspqueue_read(...)` `:1536`.
- **op batching**: 세션이 `op batching: n-ops 1024` 로 다수 op를 한 배치로 DSP에 던짐 (session hwinfo 로그, push/pop `:1356/:1427`).

→ **HTP는 op 단위 시간을 host에 주지 않고 batch/subgraph 단위로만 준다.** 그러므로 (지시대로) node 시간을 임의 분할하지 말 것. HTP 구간은 "split(=DSP batch) 실측 시간 + 포함 node 목록"으로만 기록한다. 세분 프로파일이 필요하면 `GGML_HEXAGON_PROFILE=1/2` (ggml-hexagon.cpp `:4243` 계열, per-op usec/cycle을 DSP가 리턴) 를 별도로 병기.

---

## 5. Prefill / Decode phase 구분

- ubatch token 수로 구분: prefill = `n_tokens = min(n_ctx, n_ubatch)` (대량), decode = `n_tokens == 1`. `src/llama-context.cpp:456`, graph_reserve 호출부 `:611-648`.
- ggml-backend.cpp 레벨에는 phase 정보가 직접 없음 → **graph_compute 호출 카운터**로 구분 권장: `ggml_backend_sched_graph_compute(_async)` `:1927/:1933` 진입 시 전역 카운터 증가. reserve/warmup 제외 후 **첫 실행 = prefill, 이후 = decode step N**. (decode_step, graph_index 기록.)

---

## 6. Node 이름 → block/logical-layer 매핑

- llama.cpp graph node/tensor 이름 규칙: weight = `blk.<N>.<role>.weight` (예 `blk.5.ffn_down.weight`), 중간 텐서 = `<name>-<layer>` (예 `ffn_out-5`, `Qcur-3`, `attn_norm-12`), 임베딩 = `token_embd.weight` / `inp_embd`, 최종 = `output_norm.weight`, LM Head = `result_output` (tied 모델은 `token_embd.weight`를 MUL_MAT 재사용).
- 파싱 규칙: `blk\.(\d+)\.` → block_index; suffix `-(\d+)` → layer; 그 외 embedding / output_norm / lm_head 로 분류.
- op_type: `node->op` (`GGML_OP_GET_ROWS`, `GGML_OP_MUL_MAT`, `GGML_OP_MUL_MAT_ID`, `GGML_OP_ROPE`, `GGML_OP_SOFT_MAX`, `GGML_OP_RMS_NORM`, `GGML_OP_ADD` 등). op 이름 `ggml_op_desc(node)`.

---

## 7. GGML_LAYER_PROFILE hook 설계 (위 근거 기반)

| 모드 | hook | 기록 |
| --- | --- | --- |
| **assignment** (`GGML_LAYER_PROFILE=assignment`) | `compute_splits` `:1553` 진입 시 split·node 순회 (강제 sync 없음) | run_id, phase, graph_index, split_index, node_index, node_name, op_type, weight/src/dst tensor, type/shape/bytes, **assigned_backend** (`get_tensor_backend`), **buffer_backend** (node->buffer->buft), supports_cpu/htp (`supports_op` per backend) |
| **timing** (`GGML_LAYER_PROFILE=timing`) | `compute_splits` split 별 copy/compute/sync 구간 (`:1611/:1722/:1750`) 을 timestamp 래핑 | split_index, **executed_backend** (split_backend 이름), 포함 node 목록, dispatch_us, compute_us, synchronize_us, copy_us, total_us |

- 출력: env `GGML_LAYER_PROFILE_OUT`(파일 경로)로 CSV append. 미설정 시 stderr.
- phase/decode_step: graph_compute 카운터(`:1927`) 전역 변수.
- overhead: assignment 모드는 순회만(무시 가능), timing 모드는 split 경계 sync가 추가되어 **absolute latency 왜곡 있음 → 상대 비교용**으로만 사용(지시대로 결과에 명시).
- HTP subgraph 세부는 필요 시 `GGML_HEXAGON_PROFILE`로 병기.

---

## 8. 확인된 파일 목록 (수정 후보)

| 파일 | 역할 | 수정 필요 |
| --- | --- | --- |
| `ggml/src/ggml-backend.cpp` | 스케줄러(split/assign/execute/copy/sync) | **주 계측 지점** (compute_splits, graph_compute 카운터) |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | HTP dispatch/RPC | (선택) HTP batch 시간 병기용, 기존 VERBOSE/PROFILE로 대체 가능 |
| `src/llama-context.cpp` | phase(n_tokens) 근원 | (선택) phase 태깅을 명시적으로 넘길 경우 |

> 결론: **핵심 계측은 `ggml-backend.cpp`의 `compute_splits` 한 곳**에서 assignment+timing을 모두 얻을 수 있고, backend-agnostic(HTP/CPU 공통)이라 프로덕션 실행 구조를 거의 바꾸지 않는다.
