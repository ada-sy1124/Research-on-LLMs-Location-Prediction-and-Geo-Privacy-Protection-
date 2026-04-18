# GeoAI Scripts Refactor

本目录已重构为“脚本入口 + src 包”的标准工程结构，在不改变原有功能的前提下，把公共逻辑、配置和工具函数进行了模块化。

## 目录结构

- `src/geoai_pipeline/config.py`: 统一读取 `.env` 配置
- `src/geoai_pipeline/constants.py`: Prompt 与类别映射常量
- `src/geoai_pipeline/tools/`: 通用工具函数（地理距离、Gemini 调用、数据集 IO、解析等）
- `src/geoai_pipeline/pipelines/`: 主流程脚本实现
- `src/geoai_pipeline/pipelines/helpers/`: 辅助脚本实现
- 旧文件名脚本：保留为兼容入口（可继续按原方式运行）

## 环境变量

1. 复制 `code/.env.example` 为 `code/.env`
2. 路径默认使用相对路径，且统一放在 `./data` 下（按你的本地数据位置微调并填写密钥）

## 运行方式（与旧版一致）

在 `code/` 目录下直接运行原脚本名，例如：

```bash
python 最新筛数据集_gemini从0构建YESandNO.py
python 从YES构建afterSAM.py
python gemini_sam从afterSAM构建YES_Mask.py
python 训练集文本构建.py
python 辅助功能/合并数据.py
```

## 说明

- 功能逻辑保持不变，主要是拆分结构与去硬编码。
- 所有配置默认值仍沿用原脚本常量，确保迁移后行为一致。
