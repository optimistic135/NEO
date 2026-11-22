# Online IR · PEA 与 PLEAD

| 数据集 | 子集数 | 每子集图像数 | IR | 方法 |
| --- | ---: | ---: | --- | --- |
| CIFAR-10-C | 10 | 1,000 | 1, 20, 50, 5000 | PEA, PLEAD |
| CIFAR-100-C | 100 | 100 | 1, 200, 500, 50000 | PEA, PLEAD |
| ImageNet-C | 1,000 | 100 | 1, 1000, 5000, 500000 | PLEAD |

ViT-B/16，15 个标准 corruption，severity 5，种子 1234、2020、9999。每个类别主导一个子集。按照目标概率计算四舍五入后的主导类整数配额，剩余配额在少数类中以循环方式均摊。种子确定类别排列、子集顺序、抽样索引及子集内部顺序。CIFAR 两组复用现有全流无放回构造。ImageNet-C 每个子集独立无放回抽图，独立子集之间允许复用图像，整个 100,000 样本的流只有 50,000 张不同的候选图。

适应状态按 corruption、IR、seed、方法初始化。流内不因切换子集重置。每个子集用 64 张分批，尾批单独执行。当前批参与统计后即时评分，无冻结验证集，标签只用于构造索引和评分。PLEAD 使用已有最终版的 128 样本 FIFO 与更新规则。PEA 调用固定提交的官方 ViT 实现，保留两遍预测和自身的熵触发重置机制；源统计来自干净 CIFAR，不读取 ImageNet-1K。两种方法共享模型和抽样索引。

## ImageNet-C 的容量限制

每个 corruption 有 1,000 类，每类 50 张。IR=1 的 100 个位置按循环分给类别，类别支持集独立于名义主导类别。全流类别数量均衡，单个子集不能同时包含 1,000 类。IR=5000 和 500000 分别需要 83 和 100 张不同的主导类图片。默认 `strict` 标记后两档为容量阻塞。IR=1000 的理论比例约 50.025%，四舍五入到实际 50%。

实验性 `IR_CAPACITY_POLICY=cap` 将超出容量的主导类配额截为 50，余量分给其他类，仍保留每个子集 100 张不同的图片。相关结果单独保存，属于容量截断设置。相同种子下所有取到相同整数配额的 ImageNet-C IR 会生成相同索引，不能把这些档位当作不同的实测不均衡强度。

## 运行

使用之前 balanced-512 安装的 Python 环境、固定权重缓存和 PEA 官方 checkout。入口提供数据集参数 `all`、`cifar10-c`、`cifar100-c` 或 `imagenet-c`。默认读 `/data/user3/chenc/datasets/CIFAR-clean` 和 `/data/user3/chenc/checkpoints/Vit`，可通过 `--cifar-source` 和 `VIT_CHECKPOINT_ROOT` 覆盖。默认只读权重缓存。可用 `IR_PYTHON` 指定已经装好依赖的 Python。

```bash
CUDA_VISIBLE_DEVICES=0 bash ir-benchmark/run_server.sh all
```

先执行采样单元测试。每个数据集的每个可行 IR 在 gaussian_noise / seed 2020 上先测试两个子集。通过检查的方法才进入正式评估。终端显示进度，同时写日志。

输出位于项目根目录的 `results/ir-benchmark/{strict,cap}/`。该目录下 `report.md` 汇总 Top-1、配对差和置信区间、时延及 RAM/VRAM。`summary.csv` 和 `per_stream.csv` 保存汇总与逐流数据。`full/splits/` 保存真实抽样索引、类别配额和主导类顺序；`full/configuration/` 保存模型校验和、配置、版本及代码哈希。Smoke Test 位于独立的 `smoke/`。正在计算时可用 `ir-benchmark/report.py results/ir-benchmark/strict` 手工刷新报告。
