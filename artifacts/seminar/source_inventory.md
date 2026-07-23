# Source Inventory — 근거 목록

세미나 HTML의 각 주장에 대한 저장소 근거. 경로는 `llamacpp-aos-benchmark/` 기준. 라인 번호는 checkout(llama.cpp commit `e6b88e584`, build 9793) 기준으로 직접 확인함. 인터넷 자료 미사용 — 저장소·artifact·로그·사용자 제공 수치만 사용.

## 1. 벤치마크 원본
- `artifacts/quantization/quant_matrix/quant_matrix.csv` — 12행. HTML `ROWS`에 그대로 반영(보정 없음).
- `artifacts/quantization/quant_matrix/prompt.txt` — 입력 prompt(255 token target).
- `artifacts/quantization/quant_matrix/raw/{F16,Q8_0,Q4_0,Q4_K_S,Q4_K_M,Q4_K_pure}_{HTP0,CPU}_alloc.log` — 로딩/배치 로그.

## 2. 모델 사실 (exaone4 / tied)
- arch=`exaone4`, name=`260526_ixi_GEN_1.2B_vocab_trim`, block_count=30, n_layer=30, embedding_length=2048, vocab_size=65536, tokenizer=gpt2(BPE), 333 tensors, context_length(meta)=32768
  → `artifacts/quantization/quant_matrix/raw/Q4_0_HTP0_alloc.log` (llama_model_loader kv / print_info).
- **Tied embedding 확정**:
  - `output.weight`/`lm_head.weight` tensor 부재 (only `token_embd.weight` + per-block `attn_output.weight`) → `artifacts/quantization/q4k_compare/tensor_type_comparison.csv`.
  - `token_embd.weight`가 MUL_MAT 입력으로 vocab logits(2048:65536 → 65536:512) 생성 → `Q4_0_HTP0_alloc.log`.
  - `third_party/llama.cpp/src/llama-quant.cpp:181` `bool has_tied_embeddings = true; // assume tied until we see output.weight`
  - `:100-106` `tensor_name_match_token_embd` / `tensor_name_match_output_weight`
  - `:303` `quantize &= params->quantize_output_tensor || name != "output.weight";`

## 3. 메모리 / repack (확정)
- Q4_0 CPU-only: `CPU_REPACK = 678.75 MiB` → `raw/Q4_0_CPU_alloc.log` (`load_tensors: CPU_REPACK model buffer size = 678.75 MiB`).
- Q4_0 HTP0: `HTP0-REPACK = 573.75 MiB`, `CPU_REPACK = 105.00 MiB`, `HTP0 model = 0.50 MiB` → `raw/Q4_0_HTP0_alloc.log`.
  - 573.75 + 105.00 = 678.75 (동일). 84.5% = 573.75/678.75.
- Q4_K_pure HTP0: `CPU model = 72.00`, `CPU_REPACK = 645.75`, `HTP0 model = 0.50 MiB` → `raw/Q4_K_pure_HTP0_alloc.log`.

## 4. GET_ROWS / MUL_MAT supports-op (확정 = capability, not execution trace)
- `raw/Q4_0_HTP0_alloc.log`:
  - `ggml-hex: HTP0 supports-op GET_ROWS|token_embd.weight x inp_tokens -> embd|... q6_K x i32 -> f32 ... |no`
  - `ggml-hex: HTP0 supports-op MUL_MAT|token_embd.weight x  -> |2048:65536 ... -> 65536:512|q6_K ... HTP0-REPACK ... |no` (tied LM Head)
  - `ggml-hex: HTP0 supports-op MUL_MAT|blk.0.attn_q.weight ... q4_0 ... HTP0-REPACK ... |yes` (Q4_0 blk weight = 지원)
- (참고) OPTRACE/EMBTRACE도 존재: `[OPTRACE][CPU] op=GET_ROWS ... src0_buft=CPU`, `[EMBTRACE][LOAD] token_embd.weight ... selected_buft=CPU` — 다만 전체 node별 executed-backend map은 미완, 추가 계측 대상.

## 5. Q4_K recipe (use_more_bits)
- `third_party/llama.cpp/src/llama-quant.cpp:417-418`:
  ```
  auto use_more_bits = [](int i_layer, int n_layers) -> bool {
      return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
  };
  ```
  n_layers=30 → 선택 layer = {0,1,2,5,8,11,14,17,20,23,26,27,28,29} (14개). 파이썬 재현 확인.
- Q4_K_S vs Q4_K_M 차이 = 29 tensor (attn_v 15 + ffn_down 14), M은 Q6_K로 승격; token_embd는 S·M 모두 Q6_K
  → `artifacts/quantization/q4k_compare/aggregates.json` (`diff_counts.s_vs_m=29`, `m_promotions.to_Q6_K=29`, `s_promotions.to_Q5_K=7,to_Q6_K=1`), `promoted_tensors.csv`, `tensor_type_comparison.csv`, `layer_type_summary.csv`.
- GGUF 크기: pure 680,138,976 / s 718,412,000 / m 748,804,320 bytes → `aggregates.json`.

## 6. 단말 / HTP / 빌드
- `artifacts/device_probe/20260629_162940/getprop.txt`:
  `ro.product.model=SM-F956N`, `ro.soc.model=SM8650`, `ro.board.platform=pineapple`, `ro.soc.manufacturer=QTI`, `ro.build.version.release=14`.
- HTP **V75 (candidate)**: `qnn_libs.txt`(`libQnnHtpV75Skel.so` 등) / `device_info.md`(candidate 명시).
- llama.cpp `build = 9793 (e6b88e584)` → `raw/*_alloc.log` (llama_print_build_info).

## 7. 확인하지 못한 항목 (HTML 내 "확인 필요"로 표기)
- graph node별 실제 assigned/executed backend (전체 map)
- Transformer block별·LM Head의 실제 실행 backend와 시간
- CPU↔HTP transition 횟수 / copy·dispatch·sync 시간
- cold TTFT vs warm TTFT 분리
- CSV `decode_tps` ↔ `1000/decode_ms_per_tok` 불일치 원인, `prefill_ms×prefill_tps` token 수 불일치 원인
- SoC 세대명(프로젝트 목표의 "Gen4" vs 확인된 SM8650) — 실단말 확인값 우선 표기
