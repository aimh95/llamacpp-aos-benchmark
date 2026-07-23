# pure Q4_K 단말 벤치마크 (SM-F956N / SM8650 / HTP v75)

- 단말: Samsung SM-F956N, adb serial `R3CX403LWAB`
- 실행: `llama-bench -p 128 -n 64` (pp128 = 프롬프트 처리, tg64 = 토큰 생성)
- 패키지: `/data/local/tmp/llama.cpp` (GGML_HEXAGON=ON, HTP v75), build `e6b88e584 (9793)`
- 비교 모델: `ixi_gen_1p2b_q4_k_pure.gguf`(646 MiB) vs `ixi_gen_1p2b_q4_0.gguf`(679 MiB)

## 결과

| model | config | pp128 (t/s) | tg64 (t/s) |
| --- | --- | ---: | ---: |
| **pure Q4_K** | HTP0 (`-ngl 99 --device HTP0`) | ~184 | ~24 |
| **pure Q4_K** | CPU (`-ngl 0`) | ~10 | ~37 |
| Q4_0 | HTP0 (`-ngl 99 --device HTP0`) | **~921** | ~21 |
| Q4_0 | CPU (`-ngl 0`) | ~179 | ~41 |

(원본 로그: `bench_pure_q4k_htp0.log`, `bench_matrix.log`)

## 핵심 결론

1. **HTP는 Q4_K matmul을 지원하지 않는다 (op 수준 확정).**
   verbose 로그: `supports-op MUL_MAT|blk.0.attn_q.weight ... q4_K ... HTP0-REPACK ... no`.
   → Q4_K 모델은 HTP0에 올려도 matmul이 HTP로 offload되지 않는다.

2. **프롬프트 처리(pp)가 이를 그대로 보여준다.**
   HTP0에서 Q4_0는 **921 t/s**로 HTP matmul 가속을 받지만, Q4_K는 **184 t/s**에 그친다(약 5배 차이).
   Q4_0의 pp 가속(HTP 179→921)이 HTP의 존재 이유인데, Q4_K는 그 혜택을 전혀 못 받는다.

3. **토큰 생성(tg)은 이 크기 모델에선 CPU가 오히려 빠르다.**
   HTP0 tg(21~24) < CPU tg(37~41). 1.2B급 소형 모델은 tg가 메모리 대역폭 바운드라,
   HTP의 토큰당 DSP RPC/오케스트레이션 오버헤드가 이득을 상쇄한다. (tied token_embd/output이 CPU에
   남는 것도 tg 병목에 기여.)

4. **Q4_K는 CPU pp도 느리다(~10 t/s).**
   이 ARM 빌드의 Q4_0 경로는 int8/dotprod REPACK 커널로 pp가 빠른데(CPU 179), K-quant는 그런
   최적화 pp 커널이 없어 CPU pp가 급락한다.

## 권고

이 Snapdragon HTP 디바이스 타깃에는 **Q4_0가 Q4_K보다 압도적으로 유리**하다:
- pp: HTP matmul 가속으로 Q4_0가 5배 빠름 (921 vs 184).
- Q4_K는 HTP offload가 안 되므로 NPU를 쓰는 의미가 사라진다.
- 품질을 위해 임베딩/출력을 올리고 싶다면 q6_K/q4_K가 아니라 HTP가 받는 타입(Q4_0)으로 유지해야 한다.
