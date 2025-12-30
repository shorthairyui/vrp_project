# 时间窗改动报告

## 主要代码位置
- `CVRP/CVRPEnv.py`
  - 新增时间窗字段（`ready_time`、`due_time`、`service_time`）的保存、缓存和缺省处理。
  - 在 `step()` 中按到达时间、等待时间、超窗量更新 `current_time`、`time_window_penalty`，并支持硬/软约束。
  - 在 VRPLIB 与随机问题加载时，将时间窗数据与坐标、需求一起扩展/缓存，支持默认值。
- `compute_unscaled_reward()` 与 `_get_reward()` 将时间窗罚分叠加在路程上，保持训练/推理一致性。
- 罚分计算细节：`time_window_penalty += tardiness_coeff * tardiness`，其中 `tardiness` 是超出 `due_time` 的时间，`tardiness_coeff` 用于放大或缩小超窗的成本（例如让超窗 1 单位时间相当于多行驶多少距离）。
- 评估时如传入显式解序列，`compute_unscaled_reward(solutions=...)` 会根据该序列重算到达时间与超窗量得到 `time_penalty`，以便离线解也按同样的规则计入时间窗成本。
- `CVRP/test_time_windows.py`
  - 覆盖性测试，验证到达时间、罚分累计与结束逻辑。
- 新增可选的“1-step lookahead reachability mask”（`use_lookahead_mask`），在解码前向前查看是否存在可达的后续节点，提前屏蔽会让剩余节点全部失效的候选动作。

## 与原始数据/数据生成的兼容性
- **原始数据仍可直接使用。** 若实例/批数据未提供时间窗字段：
  - `ready_time` 默认 0，`due_time` 默认无穷大，`service_time` 默认 0（可通过构造参数覆盖）。
  - 数据加载与坐标、需求的归一化/增强逻辑保持不变，仅多带了时间窗张量并同步增强。
- VRPLIB 路径：若文件无时间窗字段，同样应用上述默认值，不影响原有坐标与需求的读取。
- 随机问题生成：仍使用原来的 `loc`、`demand`、`depot`，未改变生成分布；时间窗张量若缺失则自动填入默认值并跟随 8-fold 数据增强。

## 训练/推理流程影响概述
- 奖励现在包含时间窗罚分（或硬约束直接终止），训练与推理的得分都会体现时间窗可行性。
- 每辆车独立维护 `current_time`，与载重、已访掩码一起进入动作可行性判断（可选硬约束时屏蔽超窗动作）。

## 如何验证
- 在安装好 PyTorch 后运行：
  - `python -m pytest CVRP/test_time_windows.py`
- 训练/推理脚本如未传入时间窗数据，可按原样运行；需要时间窗时传入 `ready_time`/`due_time`/`service_time` 字段或在 `CVRPEnv` 初始化时设置默认值。
