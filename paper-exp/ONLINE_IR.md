# Online IR baselines

运行 No Adapt、TENT、SAR、CoTTA、EATA、DeYO、LAME、NEO 和 ROID。基线使用上一轮 PEA／PLEAD 的 ViT-B/16、同一 checkpoint、severity 5、15 种扰动、三个种子及批次上限 64。采样器直接共享，CIFAR-10-C 的 IR 为 1/20/50/5000，CIFAR-100-C 为 1/200/500/50000，ImageNet-C 为 1/1000/5000/500000。子集尾批保留，状态跨子集，其重置和当前批预测语义由方法自身决定。评估标签仅用于采样和评分。

复用 TENT、SAR、CoTTA、DeYO 和 LAME 适配器。ImageNet-C 用 CoTTA 的官方 ImageNet 分支配置，CIFAR 用官方 CIFAR 分支。EATA 调用官方核心，保持 entropy/diversity 和 Fisher 正则。Fisher 用 2,000 张独立干净图和模型伪标签估计，按种子缓存于当前进程。默认仅 CIFAR 提供干净源数据，缺独立干净 ImageNet 时将完整 EATA 标为缺源阻塞。可用 --imagenet-clean 指定含 train/ 的干净 ImageNet 根目录。NEO 用作者仓库的累计特征重心；ROID 使用官方损失、先验校正、模型集成与官方增强。方法配置与来源版本见 configs/methods/。

```bash
CUDA_VISIBLE_DEVICES=0 bash paper-exp/scripts/run_online_baselines.sh
```

默认输出在 paper-exp/results/ir_baselines/strict/。严格模式沿用原容量限制，ImageNet-C IR=5000、500000 不满足每个子集 100 张不同图像与主导类配额。--capacity-policy cap 是隔离输出的容量截断实验，不能当作原高 IR 档。支持 --dataset、--methods、--smoke-only。默认先做每种方法和 IR 的两子集 smoke，再做正式评估。再次执行时校验哈希后跳过已有成功流。

report.md、summary.csv、per_stream.csv、paired_differences.csv 汇总结果。采样索引在 full/splits，逐流和逐批结果在 full/raw。历史 PEA／PLEAD 从 results/ir-benchmark/<policy> 读取。只有 checkpoint、数据顺序、测试流与评估索引的哈希全部吻合的正式结果才并入报告；有完整配对方可计算 95% CI。
