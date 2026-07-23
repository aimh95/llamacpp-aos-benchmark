# Q4_K recipe 실측 비교 — ixi_gen_1p2b

`models/ixi_gen_1p2b_f16.gguf`(EXAONE4 아키텍처, 30 layers, tied embedding)를 세 가지 방식으로
양자화한 뒤, **생성된 GGUF의 실제 텐서 타입을 직접 읽어** 비교한 결과. 양자화 로그가 아니라
GGUF 메타데이터에서 텐서별 `tensor_type`을 추출했다.

- 비교 대상: `Q4_K_S`, `Q4_K_M`, pure `Q4_K`
- 추출 스크립트: [scripts/quantization/compare_q4k_recipes.py](../../scripts/quantization/compare_q4k_recipes.py)
- 원본 CSV: [artifacts/quantization/q4k_compare/](../../artifacts/quantization/q4k_compare/)

---

## A. 실행 환경

| 항목 | 값 |
| --- | --- |
| llama.cpp commit | `e6b88e58446b72e9c11334a3f7a52fb73fd3b4a2` |
| llama-quantize build | `9793 (e6b88e584)`, GNU 11.4.0 x86_64 |
| 입력 모델 | `models/ixi_gen_1p2b_f16.gguf` (F16, 333 tensors) |
| 총 텐서 수 | 333 (양자화 대상 211 + F32 norm/1D 122) |
| tied embedding | **true** → 별도 `output.weight` 없음. `token_embd.weight`가 입력 임베딩과 출력 projection을 겸함 |

**pure Q4_K 생성 방식.** 현재 CLI에서 `Q4_K`는 독립 preset이 아니라 **`Q4_K_M`의 alias**(ftype id 15)다
(`llama-quantize --help`: `15 or Q4_K : alias for Q4_K_M`). 따라서 "모든 양자화 대상 텐서를 Q4_K로"
만드는 pure 버전은 per-tensor 승격을 끄는 `--pure` 플래그와 조합해서 생성한다:

```bash
build-host/bin/llama-quantize --pure \
  models/ixi_gen_1p2b_f16.gguf \
  models/quant_compare/ixi_gen_1p2b_q4_k_pure.gguf \
  Q4_K_M
```

`--pure`가 실제로 승격을 전부 제거하는지는 아래 C의 타입 집계로 확인된다(pure에는 Q5_K/Q6_K가 0개).

---

## B. 파일 크기 비교

| Recipe | GGUF 크기 (bytes) | ≈ MiB |
| --- | ---: | ---: |
| pure Q4_K | 680,138,976 | 648.6 |
| Q4_K_S | 718,412,000 | 685.1 |
| Q4_K_M | 748,804,320 | 714.1 |

크기 순서 pure < S < M 은 승격된 텐서 수(0 < 8 < 29)와 정확히 일치한다.

---

## C. 타입 개수 비교

| 타입 | pure Q4_K | Q4_K_S | Q4_K_M |
| --- | ---: | ---: | ---: |
| F32 | 122 | 122 | 122 |
| Q4_K | 211 | 203 | 182 |
| Q5_K | 0 | 7 | 0 |
| Q6_K | 0 | 1 | 29 |
| Q8_0 | 0 | 0 | 0 |
| F16 | 0 | 0 | 0 |

- **pure Q4_K**: 양자화 대상 211개가 전부 Q4_K. 승격 없음(= `--pure` 검증).
- **Q4_K_S**: Q5_K 7개 + Q6_K 1개 = 8개만 pure에서 상향.
- **Q4_K_M**: Q6_K 29개. **Q5_K는 하나도 안 쓴다** — M은 Q4_K 아니면 Q6_K로 이분된다.

---

## D. 실제 변경 텐서

(전체 목록: [promoted_tensors.csv](../../artifacts/quantization/q4k_compare/promoted_tensors.csv))

### pure Q4_K 대비 Q4_K_S에서 바뀐 텐서 (8개)

| 텐서 | role | pure → S |
| --- | --- | --- |
| `token_embd.weight` | token_embd | Q4_K → **Q6_K** |
| `blk.0/1/2/3.attn_v.weight` | attn_v | Q4_K → **Q5_K** (레이어 0,1,2,3) |
| `blk.0/1/2.ffn_down.weight` | ffn_down | Q4_K → **Q5_K** (레이어 0,1,2) |

→ token_embd 1개(Q6_K) + attn_v 4개(Q5_K) + ffn_down 3개(Q5_K).

### pure Q4_K 대비 Q4_K_M에서 바뀐 텐서 (29개)

| 텐서 | role | pure → M | 레이어 |
| --- | --- | --- | --- |
| `token_embd.weight` | token_embd | Q4_K → **Q6_K** | — |
| `blk.N.attn_v.weight` | attn_v | Q4_K → **Q6_K** | 0,1,2,5,8,11,14,17,20,23,26,27,28,29 (14개) |
| `blk.N.ffn_down.weight` | ffn_down | Q4_K → **Q6_K** | 0,1,2,5,8,11,14,17,20,23,26,27,28,29 (14개) |

→ token_embd 1 + attn_v 14 + ffn_down 14 = 29. 전부 Q6_K(Q5_K 미사용).

### Q4_K_S 대비 Q4_K_M에서 추가로 바뀐 텐서 (29개)

`token_embd`는 S/M 모두 Q6_K로 **동일**(변화 없음). 나머지 attn_v/ffn_down에서 29개가 갈린다:

- **attn_v (15개 상이)**: 레이어 0,1,2 → S:Q5_K vs M:Q6_K; 레이어 3 → S:Q5_K vs M:Q4_K; 레이어 5,8,11,14,17,20,23,26,27,28,29 → S:Q4_K vs M:Q6_K
- **ffn_down (14개 상이)**: 레이어 0,1,2 → S:Q5_K vs M:Q6_K; 레이어 5,8,11,14,17,20,23,26,27,28,29 → S:Q4_K vs M:Q6_K

15 + 14 = **29개**.

---

## E. 레이어별 실질 변화

**변화가 전혀 없는 텐서(전 레이어·전 recipe 고정):**

- `attn_q`, `attn_k`, `attn_output`, `ffn_gate`, `ffn_up` → **30개 레이어 모두, 세 recipe 모두 Q4_K** (5 × 30 = 150 텐서, 승격 0)
- 모든 norm/1D 텐서(`*_norm`, `output_norm`) → F32
- 즉 세 recipe의 차이는 **오직 `attn_v`, `ffn_down`, 그리고 `token_embd`** 세 종류에서만 발생한다.

**변화가 있는 텐서(attn_v, ffn_down)의 레이어별 타입:**

| layer | attn_v (pure/S/M) | ffn_down (pure/S/M) |
| ---: | --- | --- |
| 0 | Q4_K / Q5_K / **Q6_K** | Q4_K / Q5_K / **Q6_K** |
| 1 | Q4_K / Q5_K / **Q6_K** | Q4_K / Q5_K / **Q6_K** |
| 2 | Q4_K / Q5_K / **Q6_K** | Q4_K / Q5_K / **Q6_K** |
| 3 | Q4_K / **Q5_K** / Q4_K | Q4_K / Q4_K / Q4_K |
| 4 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 5 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 6 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 7 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 8 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 9 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 10 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 11 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 12 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 13 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 14 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 15 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 16 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 17 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 18 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 19 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 20 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 21 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 22 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 23 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 24 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 25 | Q4_K / Q4_K / Q4_K | Q4_K / Q4_K / Q4_K |
| 26 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 27 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 28 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |
| 29 | Q4_K / Q4_K / **Q6_K** | Q4_K / Q4_K / **Q6_K** |

**승격 레이어 패턴 (M):** attn_v와 ffn_down 모두 **동일한 14개 레이어 집합**에서 Q6_K가 된다:
`{0,1,2}` (첫 부분) ∪ `{5,8,11,14,17,20,23}` (중간, 3칸 간격) ∪ `{26,27,28,29}` (마지막 부분).
이는 llama.cpp의 `use_more_bits` 규칙(`i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2`, n=30)과 정확히 일치한다.

**S 패턴:** 첫 부분만 얕게 올린다 — attn_v는 레이어 0,1,2,3, ffn_down은 레이어 0,1,2를 Q5_K로.

---

## F. 최종 결론 (실측 GGUF 기준)

1. **Q4_K_S vs pure Q4_K:** 8개 텐서만 다르다 — `token_embd`(Q4_K→Q6_K), `attn_v` 레이어 0·1·2·3(→Q5_K),
   `ffn_down` 레이어 0·1·2(→Q5_K). 나머지 203개 양자화 텐서는 pure와 동일한 Q4_K.

2. **Q4_K_M vs pure Q4_K:** 29개 텐서가 다르다 — `token_embd`(→Q6_K), `attn_v` 14개 레이어(→Q6_K),
   `ffn_down` 14개 레이어(→Q6_K). M은 Q5_K를 전혀 쓰지 않고 Q4_K↔Q6_K 이분.

3. **M이 레이어 전체를 고정밀로 바꾸는가? 아니다.** 승격은 각 레이어의 `attn_v`와 `ffn_down` **두 텐서에만**
   적용된다. `attn_q`/`attn_k`/`attn_output`/`ffn_gate`/`ffn_up`은 승격된 레이어에서도 Q4_K 그대로다.

4. **어떤 레이어의 어떤 텐서가 Q5_K/Q6_K가 되는가:** 위 E 표 그대로. 핵심만: M은 14개 선택 레이어의
   attn_v·ffn_down을 Q6_K로, S는 앞쪽 3~4개 레이어의 attn_v·ffn_down을 Q5_K로, 그리고 두 recipe 모두
   token_embd를 Q6_K로.

5. **첫/중간/마지막 승격 패턴이 실제로 관찰되는가? 그렇다(M).** M의 승격 레이어 `{0,1,2}`(첫),
   `{5,8,11,14,17,20,23}`(중간 3칸 간격), `{26,27,28,29}`(마지막)은 `use_more_bits` 패턴과 일치한다.
   레이어 3·4·6·7·9·10…은 승격되지 않는다. S는 앞부분만 얕게 승격.

6. **Q4_K_S ↔ Q4_K_M 사이에서 달라지는 텐서 수: 29개** (attn_v 15 + ffn_down 14; token_embd는 양쪽 Q6_K로 동일).
   그중 M에서 Q6_K로 승격되는 텐서는 29개(→Q5_K 0개), S에서 승격되는 텐서는 8개(→Q5_K 7개, →Q6_K 1개).

---

## 산출물

| 파일 | 내용 |
| --- | --- |
| [tensor_type_comparison.csv](../../artifacts/quantization/q4k_compare/tensor_type_comparison.csv) | 333개 텐서 전부 × 세 recipe 타입 + 변경 플래그 |
| [layer_type_summary.csv](../../artifacts/quantization/q4k_compare/layer_type_summary.csv) | 레이어별 7개 proj 역할 타입(recipe별 컬럼) |
| [promoted_tensors.csv](../../artifacts/quantization/q4k_compare/promoted_tensors.csv) | 어느 recipe에서든 pure와 달라진 텐서 30행(token_embd 1 + attn_v 15 + ffn_down 14) + change_pattern |
| [aggregates.json](../../artifacts/quantization/q4k_compare/aggregates.json) | 크기·타입 집계·정합성 검증 |
| [compare_q4k_recipes.py](../../scripts/quantization/compare_q4k_recipes.py) | 추출·비교 스크립트 |
