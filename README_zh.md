# MotionScore 肩部上举动作分析

本仓库提供与 2026-10-21 第四轮论文版本对应的独立研究代码：
视频 → 身体关键点 → BiLSTM–CRF 四阶段 → 13 项运动学特征 → 映射 → 分项偏离、MotionScore 和参考百分位。

## 可复现范围

- 316 人：242 人开发集和 74 人比较集，固定受试者级五折。
- 13 项特征、原始双侧幅度惩罚、冻结块预算、精确阈值。
- Identity、Ridge、ElasticNet、LightGBM 四个配置；包含全开发模型及五折模型。
- 冻结推理、跨周期预测指标和映射器重新训练。
- 相位模型的训练程序和伪标签。神经网络跨硬件重新训练不承诺逐位一致。

其余深度表格基准、T-JEPA 编码器训练、体测档案关联和硬件延迟实验不在本发布包的已验证复现范围内。
公共阈值保留历史 T-JEPA 开发分支的原数值，不能换后端后重新调阈值。

## 安装和运行

仓库内含一个可直接运行的 Ridge 小样例，不需要下载完整权重和数据，也不需要 GPU：

```bash
python -m pip install -e .
motionscore score-pose --input examples/example_pose.csv --labels examples/example_labels.npz --models examples/models --backend Ridge --output results/example.json
```

结果可与 `examples/example_expected.json` 对比。完整复现流程如下。

在 Python 3.10 新环境中，按英文 [README](README.md) 安装。

```bash
python -m pip install -e ".[phase,trees,train,test]"
python scripts/download_release.py --kind models
python scripts/download_release.py --kind dataset
python -m pytest -q
python scripts/verify_release.py --data downloads/dataset-v1.0.0 --models downloads/models-v1.0.0 --phases
```

脚本会读取原程序生成的金标准结果，比较特征、分数、分档，并输出折外 MSE、ICC。
详细运行顺序与验证限度见 [复现指南](docs/REPRODUCIBILITY.md)。

## 视频和隐私

发布视频保留房间与器械背景，采用不透明头部遮挡，并对额外出现的背景人员局部遮挡；删除音轨，并清理原文件名、录制时间、身份字段。
关键点、标注与视频共用随机研究编号，原身份映射不发布。

遮挡视频重新提取关键点可能改变结果。论文数值复现以附带的原分析身体关键点为输入。
骨架和衣着仍可能保留可识别线索，因此本项目不声称达到不可重新识别的完全匿名。

MotionScore 描述特定任务下相对参考群体的运动学偏离；当前证据不支持临床诊断、一般体质预测或专家质量评分效度。

代码和本项目训练权重采用 Apache-2.0；研究数据采用 CC BY 4.0。
第三方代码及姿态预训练权重适用各自许可证。
