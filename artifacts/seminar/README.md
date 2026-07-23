# EXAONE 온디바이스 추론 세미나 자료

`exaone_htp_quant_seminar.html` — EXAONE 1.2B(exaone4)의 양자화별 CPU·HTP 실행 구조와 성능을 다루는 단일 HTML 발표 자료. 외부 CDN/네트워크 없이 단독 실행된다(순수 inline SVG 차트, Chart.js 등 미사용).

## 여는 방법
- 파일을 브라우저에서 그대로 열면 된다: `xdg-open exaone_htp_quant_seminar.html` (또는 더블클릭).
- 별도 웹서버 불필요. 인터넷 연결 불필요.

## 발표 모드 사용법
- 상단 우측 **⛶ 전체화면** 버튼 → 발표용 fullscreen.
- 상단 sticky **목차**로 섹션 이동, 현재 섹션은 자동 강조.
- 각 주요 섹션 하단 **🎤 발표자 노트**(회색 점선 박스)를 클릭해 펼치기/접기. (인쇄 시 자동 숨김)
- 차트의 **Quant 필터**와 **HTP0/CPU 토글**로 표시 대상을 바꿀 수 있고, **원본 데이터 표**는 헤더 클릭으로 정렬된다. 차트 막대에 마우스를 올리면 tooltip 표시.
- 핵심 용어(점선 밑줄)에 마우스를 올리면 쉬운 뜻 tooltip.

## 데이터 출처
- 벤치마크 원본: `artifacts/quantization/quant_matrix/quant_matrix.csv` (12행, 임의 보정 없이 그대로 HTML JS `ROWS`에 반영).
- 메모리/배치/GET_ROWS 로그: `artifacts/quantization/quant_matrix/raw/*_alloc.log`.
- Q4_K recipe 비교: `artifacts/quantization/q4k_compare/{aggregates.json,promoted_tensors.csv,tensor_type_comparison.csv,layer_type_summary.csv}`.
- 단말/HTP 정보: `artifacts/device_probe/20260629_162940/`.
- 소스 근거 라인: `third_party/llama.cpp/src/llama-quant.cpp` (commit e6b88e584 / build 9793).
- 전체 근거 표는 HTML **13장 근거 및 출처**와 `source_inventory.md` 참조.

## 차트 수치 수정 방법
- HTML 하단 `<script>` 안의 `const ROWS = [...]` 배열이 유일한 데이터 원본이다. quant_matrix.csv가 갱신되면 이 배열의 해당 행만 바꾸면 모든 차트/표가 자동 갱신된다(별도 빌드 불필요).
- 필드명은 CSV 헤더와 동일: `quant, backend, htp_layers, cpu_layers, htp_mib, cpu_mib, ttft_ms, prefill_ms, prefill_tps, decode_ms_per_tok, decode_tps`.

## 인쇄 / PDF 변환
- 브라우저 인쇄(Ctrl/Cmd+P) → "PDF로 저장". `@media print`가 목차·버튼·발표자 노트·tooltip을 숨기고 섹션이 페이지 경계에서 잘리지 않도록 처리한다.

## 확인된 사실 vs 아직 미확정 (요약)
**확정:** ① F16/Q8_0/Q4_0은 HTP에 대형 weight buffer 생성 ② Q4_0 HTP0-REPACK 573.75 MiB(=CPU-only 678.75의 84.5%) ③ Q4_K는 htp_layers=31이어도 HTP weight buffer 0.50 MiB(대형 미생성) ④ Q4_0 Prefill HTP≈3.49× 빠름 ⑤ 모든 quant에서 Decode는 CPU-only가 더 빠름 ⑥ GET_ROWS(token_embd, tied)는 HTP0 supports-op=no.

**미확정(추가 profiling 필요):** node별 실제 executed backend, LM Head 실행 backend, backend boundary transition 횟수·시간, layer별 latency, cold/warm TTFT 분리, CSV TPS↔ms/token 불일치 원인.

## 향후 layer profiling 결과를 HTML에 추가하는 위치
- **12장(다음 개발 계획)** 아래에 "우선순위 1: Layer/Op backend map" 결과 표를 넣을 자리를 마련해 두었다. 권장 데이터 schema:
  ```
  layer_backend_map: [{ quant, phase("prefill"|"decode"), node("embedding"|"blk.N.attn"|...|"lm_head"),
                        assigned_backend("CPU"|"HTP0"), executed_backend, ms }]
  ```
- 새 배열을 `<script>`의 `ROWS` 아래에 추가하고, `renderCharts()` 스타일의 `drawBars()` 호출을 한 번 더 만들면 기존 차트 엔진을 그대로 재사용할 수 있다(추가 라이브러리 불필요).
