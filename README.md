# VeReMi NextGen highway_2 数据预处理

把 VeReMi NextGen 数据集 highway_2 场景（1 个 baseline + 15 种攻击）的原始 JSON 消息，整理成可以直接用于机器学习的表格。

## 数据说明

- 数据集：VeReMi NextGen，highway_2 场景（高速公路、低密度车流）
- 原始结构：外层 ZIP → Train / Validation / Test → 内层 ZIP → 每辆接收车一个 JSON 文件
- 样本单位：一行 = 一条被接收车辆收到的 CAM 消息

## 功能

- 递归读取嵌套 ZIP 和 JSON，无需手动解压
- 展平 receiver / sender 嵌套字段，把 `"x,y,z"` 坐标字符串拆成数值列
- 派生字段：`delay_ns`（收发时间差）、`sender_receiver_distance_m`（发信车到收信车的直线距离）
- 自动生成标签：`is_attack`（是否攻击）、`attack_type`（攻击类型或 benign）
- 输出完整 Parquet 主表、分层抽样 CSV 预览、逐分区处理日志和处理摘要
- 不删除攻击造成的异常值，不做缺失值填补和标准化（这些信号可能正是模型要学习的）

## 标签规则

| 情况 | is_attack | attack_type |
|---|---|---|
| 攻击子集内 attacker=1 | 1 | 该攻击子集名 |
| 攻击子集内 attacker=0 | 0 | benign |
| baseline（无 attacker 字段） | 0 | benign |

另用 `attacker_source` 列记录标签来源（`message_field` / `baseline_assigned_zero` / `missing_in_message`）。

注意：不能按压缩包名直接打标签——攻击子集内大部分消息仍是正常车辆发送的。

## 环境依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python highway2_数据处理脚本.py --input-dir <存放外层ZIP的目录> --output-dir <输出目录>
```

可选参数：

- `--additional-zip`：额外补充的 ZIP 路径，可重复传入
- `--preview-per-part`：每个分区抽入预览 CSV 的行数（默认 250）
- `--work-dir`：中间文件目录（默认使用系统临时目录）

## 输出文件

| 文件 | 说明 |
|---|---|
| highway2_机器学习主表.parquet | 完整主表（1,733,541 行 × 49 列） |
| highway2_数据预览.csv | 每个分区抽样 250 行，便于快速查看 |
| highway2_处理日志.csv | 每个子集 × split 的文件数、行数、标签分布、解析错误数 |
| highway2_处理摘要.json | 总行数、标签总量等汇总信息 |

## 单文件示例

`单文件处理练习.py` 演示最小处理流程：读取一个 JSON → 展平全部消息 → 打标签 → 存成 CSV。适合用来理解核心逻辑。

## 攻击分布可视化

`plot_attack_distributions.py` 基于主表生成 15 类攻击的"攻击 vs 同包正常"分布对比图（KDE 密度曲线 / 直方图 / 数量统计），并输出 15 合 1 总览图。

```bash
python plot_attack_distributions.py --input highway2_机器学习主表.parquet --output-dir 攻击分布对比
```

结论摘要：11 类攻击在分布图上明显可分；suddenConstantSpeed 部分可见；positionMirroring、timeDelayAttack、dataReplay 在单条消息分布上不可分，需要相邻消息级别的时序特征。
