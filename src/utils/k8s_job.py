from __future__ import annotations
from textwrap import dedent

def generate_k8s_job(name: str = "train-job", image: str = "repo/ai-trading:latest", gpus: int = 1, command: str = "python cli.py train AAPL --advanced") -> str:
    return dedent(f"""
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: {name}
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: trainer
            image: {image}
            resources:
              limits:
                nvidia.com/gpu: {gpus}
            command: ["/bin/bash","-c","{command}"]
    """)
