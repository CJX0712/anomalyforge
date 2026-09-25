# AnomalyForge 容器镜像 — 作者：晨星
FROM python:3.13-slim

WORKDIR /app

# 系统依赖（PyOD 编译依赖极少，slim 足够）
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认运行 demo
CMD ["python", "-m", "anomalyforge.examples.run_demo"]
