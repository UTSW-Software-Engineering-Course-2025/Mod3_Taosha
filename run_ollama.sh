#!/bin/sh

unset http_proxy
unset https_proxy
module load ollama/0.7.0-img 
export OLLAMA_MODELS=/project/GCRB/Hon_lab/s440862/courses/se/MODULE_3_MATERIALS/ollama/models
ollama serve &
# !${APP_IMAGE} run qwen3:4b