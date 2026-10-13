# 参与 AdCoEvo 开发

欢迎贡献代码、文档和复现说明。报告缺陷或提出较大的功能改动时，请先通过 Issue 说明问题与预期方案。

## 搭建开发环境

使用 Python 3.10。将完整的 AuctionNet 模拟器克隆到 AdCoEvo 仓库根目录；如果 AuctionNet 位于其他位置，可设置 `ADCOEVO_AUCTIONNET`：

```bash
git clone https://github.com/jd-opensource/AdCoEvo.git adcoevo
cd adcoevo
git clone https://github.com/alimama-tech/AuctionNet
conda create -n adcoevo python=3.10 -y
conda activate adcoevo
pip install -r requirements.txt
```

使用 GPU 时，请按照 CUDA 版本安装匹配的 PyTorch 和 vLLM。运行连续投放与策略更新流程还需准备兼容的因果语言模型、Planner、Optimizer、Inspector 三个角色的初始 LoRA 适配器，以及冻结的 Inspector 奖励统计量；参见 [README.zh-CN.md](README.zh-CN.md)。

## 提交贡献

1. 在独立分支完成单一主题的修改。代码按职责放入 `agents/`、`env/`、`train/` 或 `scripts/`。
2. 验证受影响的运行流程，并运行 `git diff --check`。在 Pull Request 中说明验证结果及尚未验证的部分。
3. 修改命令行参数、环境配置、输出格式或运行假设时，同步更新文档，并保持 [README.md](README.md) 与 [README.zh-CN.md](README.zh-CN.md) 一致。
4. 通过 Pull Request 提交，说明修改目的、实现内容和验证结果；修复缺陷时附上复现步骤。

不要提交 API 密钥、内部业务数据、模型权重、训练轨迹或第三方数据集。完整 AuctionNet 模拟器由使用者单独获取。引用或改写第三方代码时，请注明来源和许可证，并更新 [NOTICE](NOTICE)。本仓库的贡献遵循 [Apache License 2.0](LICENSE)。
