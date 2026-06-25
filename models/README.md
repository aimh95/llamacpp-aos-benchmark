# models/

벤치마크에 사용할 gguf 모델을 이 폴더에 배치합니다.

- 파일명: `model.gguf` (symlink 또는 실제 복사본)
- 실제 가중치 파일은 `.gitignore`에 의해 추적되지 않습니다.

```bash
ln -s /path/to/actual-model.gguf models/model.gguf
# 또는
cp /path/to/actual-model.gguf models/model.gguf
```
