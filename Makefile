# AnomalyForge Makefile — 作者：晨星
PYTHON ?= python3

.PHONY: help install venv test demo benchmark cli clean

help:
	@echo "Targets:"
	@echo "  make venv     创建虚拟环境并安装依赖"
	@echo "  make install  安装依赖"
	@echo "  make test     运行 pytest"
	@echo "  make demo     运行端到端演示"
	@echo "  make benchmark 运行跨检测器横向评测 (CLI)"
	@echo "  make cli      运行命令行入口"
	@echo "  make clean    清理缓存"

venv:
	$(PYTHON) -m venv .venv
	.venv/Scripts/activate || . .venv/bin/activate
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m anomalyforge.examples.run_demo

benchmark:
	$(PYTHON) -m anomalyforge.cli --benchmark

cli:
	$(PYTHON) -m anomalyforge.cli --detector iforest

clean:
	rm -rf .pytest_cache __pycache__ .venv
	find . -name '__pycache__' -type d -exec rm -rf {} +
