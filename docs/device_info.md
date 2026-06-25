# Device Info

측정에 사용한 기기 사양을 기록합니다.

| 항목 | 값 |
| --- | --- |
| 기기명 | |
| SoC / 칩셋 | |
| CPU 구성 (big.LITTLE 등) | |
| RAM | |
| Android 버전 | |
| 커널 버전 | |
| ABI | arm64-v8a |

```bash
adb shell getprop ro.product.model
adb shell getprop ro.board.platform
adb shell getprop ro.build.version.release
adb shell cat /proc/cpuinfo
adb shell cat /proc/meminfo
```
