# exp-auto-scheduler

다양한 모델/하이퍼파라미터 실험을 먼저 1 epoch으로 프로파일링한 뒤, 수집된 GPU 메모리 사용량·실행 시간·실패 여부를 기반으로 현재 장비에 맞춰 자동 스케줄링하는 Python 라이브러리입니다.

## 핵심 아이디어

1. **준비 단계**: 전체 실험 명령어를 `probe` 모드로 실행합니다. 보통 `{epochs}`를 `1`로 치환합니다.
2. **지표 수집**: 각 프로세스의 실행 시간, return code, stdout/stderr 로그, CPU RSS, NVIDIA GPU 메모리 피크를 기록합니다.
3. **자동 스케줄링**: probe 결과를 이용해 실험별 GPU 메모리를 추정하고, GPU별 여유 예산 안에서 가능한 실험을 병렬로 실행합니다.
4. **상태 저장/resume**: `runs/state.json`에 probe/run 결과를 저장해 중단 후 재시작할 수 있습니다.

## 설치

```bash
pip install -e .
```

NVIDIA GPU 메모리 측정은 `nvidia-smi`를 사용합니다. `nvidia-smi`가 없으면 CPU-only 모드로 동작합니다.

## YAML 예시

```yaml
scheduler:
  work_dir: runs/resnet_sweep
  max_gpu_memory_fraction: 0.90
  reserve_gpu_memory_mb: 1024
  safety_margin_mb: 768
  retry_failed_runs: 1

experiments:
  - name: "{model}_lr{lr}_bs{batch_size}"
    command: "python train.py --model {model} --lr {lr} --batch-size {batch_size} --epochs {epochs}"
    params:
      model: resnet50
      epochs: 30
    probe_params:
      epochs: 1
    grid:
      lr: [0.001, 0.0003]
      batch_size: [32, 64]
    num_gpus: 1
    priority: 10
```

실행:

```bash
exp-scheduler run experiments.yaml
```

준비 단계만 실행:

```bash
exp-scheduler prepare experiments.yaml
```

기존 probe 결과로 추정치 확인:

```bash
exp-scheduler estimate experiments.yaml
```

## Python API 예시

```python
from exp_scheduler import ExperimentScheduler, SchedulerConfig, expand_grid

experiments = expand_grid(
    name="{model}_lr{lr}_bs{batch_size}",
    command="python train.py --model {model} --lr {lr} --batch-size {batch_size} --epochs {epochs}",
    params={"model": "resnet50", "epochs": 30},
    probe_params={"epochs": 1},
    grid={"lr": [1e-3, 3e-4], "batch_size": [32, 64]},
)

scheduler = ExperimentScheduler(
    SchedulerConfig(
        work_dir="runs/resnet_sweep",
        safety_margin_mb=768,
        reserve_gpu_memory_mb=1024,
    )
)
summary = scheduler.run(experiments)
```

## 명령어 템플릿 규칙

- `{epochs}`, `{lr}`, `{batch_size}`처럼 `params`, `probe_params`, `run_params`, `grid`에 있는 값을 사용할 수 있습니다.
- `probe` 실행 시에는 `params` 위에 `probe_params`가 덮어씌워집니다.
- `run` 실행 시에는 `params` 위에 `run_params`가 덮어씌워집니다.
- 복잡한 shell 문법이 필요하면 YAML에 `shell: true`를 지정할 수 있습니다.

## 생성되는 파일

```text
runs/
  state.json                # 전체 상태 DB
  last_summary.json          # 마지막 실행 요약
  logs/
    .../
      stdout.log
      stderr.log
      metrics.json
```

## 주의사항

- 1 epoch 메모리 피크가 전체 학습 중 최대 메모리와 다를 수 있어 `safety_margin_mb`를 둡니다.
- 다른 사용자의 프로세스가 GPU를 쓰고 있으면 시작 시점의 사용량을 제외하고 보수적으로 예산을 잡습니다.
- GPU 메모리는 `nvidia-smi --query-compute-apps` 기준으로 측정하므로, 일부 드라이버/컨테이너 환경에서는 0으로 나올 수 있습니다. 이때는 `unknown_gpu_mem_mb`를 사용합니다.
