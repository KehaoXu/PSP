# SD1.5 GenEval 全量轨迹缓存

按《SD1.5 GenEval 全量缓存实验设置》（2026-10-25）实现：553 个 prompt × 32 个种子 × DDIM 32 步，存 x_T、每步 x̂₀、偶数步 x_t，偶数步对 x̂₀ 打 IR / HPS / BLIP-ITM 分并存 embedding，保存最终 PNG 并打分。

| 文件 | 作用 |
|---|---|
| `cache_lib.py` | 共用代码：采样（逐步取 x̂₀）、初始噪声、解码、三个打分器、目录布局、folds |
| `collect_cache.py` | 填缓存，一个进程处理一段 prompt（一张卡一个分片），可断点续跑 |
| `check_cache.py` | `full`：跑完后的数据检查，并合并出总表；`same`：两份缓存逐位比较（确定性检查） |
| `verify_midterm.py` | 用中期的噪声和 batch 跑本代码，核对能否复现中期的 IR |
| `geneval_io.py` | 导出给 GenEval 的目录；把 GenEval 结果补进 `final.parquet` |
| `setup_env.sh` / `env.sh` / `requirements_cache.txt` | RunPod 上的环境 |
| `smoke_test.sh` / `run_shards.sh` | 运行步骤 1–2 / 步骤 3 |

## 在 RunPod 上运行

0. 开一个 4×RTX 4090 的 pod，挂 100 GB 网络卷到 `/workspace`。把仓库放到 `/workspace/PSP`（`git clone` 之后，用 `runpodctl send` 或 scp 补上还没提交的 `exps/cache_sd15_geneval_t32/`，以及中期结果 `exps/run_collect_image_reward_signal_sdv15_colab/ir_every4_wide.csv`）。路径都在 `env.sh` 里，可以用环境变量改。
1. `bash exps/cache_sd15_geneval_t32/setup_env.sh`：在网络卷上建 Python 3.10 venv，pod 重启后不用重装。
2. `bash exps/cache_sd15_geneval_t32/smoke_test.sh`（约 15 分钟，第一次会下载全部模型），会依次检查：
   - 2 个 prompt × 32 种子：采样器与 PSP 的 `FKDStableDiffusion.__call__` 的最终 latent 一致；文件齐全；x̂₀ 重新解码、重新打分与表中一致。最后一行给出每个 prompt 的耗时，并换算成全量单卡和 4 卡的时间，以及显存峰值。
   - 在另一张卡上把 prompt 0 再跑一遍，latent、分数、PNG 必须逐位相同。
   - 用中期的噪声跑本代码，看能否复现中期的 IR（见下文"与设置文档的出入"第 1 条）。
3. `bash exps/cache_sd15_geneval_t32/run_shards.sh`：在 tmux 会话 `cache` 里开 4 个分片（prompt 0–138 / 139–276 / 277–414 / 415–552），日志在 `$CACHE_DIR/logs/`。中断后再运行一次即可续跑。
4. 全部跑完后，先 `source exps/cache_sd15_geneval_t32/env.sh`（激活 venv、设置 `$CACHE_DIR`），再运行 `python exps/cache_sd15_geneval_t32/check_cache.py full $CACHE_DIR --gpu`。检查结果写进 `checks.json`，并生成 `scores_all.parquet`（283,136 行）和 `final_all.parquet`（17,696 行）。
5. 停掉 pod。分数表和 `prompts/*/final/` 下载到本地，latent 留在网络卷上。中期的 PNG 实测每张约 0.53 MB，比设置文档估的 0.4 MB 大，所以最终图约 9 GB，整个缓存约 37 GB。
6. GenEval（检测器环境见 `setup/setup_geneval.sh`；`export` 和 `merge` 在缓存的 venv 里跑）：
   ```bash
   python exps/cache_sd15_geneval_t32/geneval_io.py export $CACHE_DIR $CACHE_DIR/geneval_view   # 本地评测加 --copy
   cd geneval && python evaluation/evaluate_images.py $CACHE_DIR/geneval_view \
       --outfile $CACHE_DIR/geneval_results.jsonl --model-path <objdet 目录>
   python exps/cache_sd15_geneval_t32/geneval_io.py merge $CACHE_DIR $CACHE_DIR/geneval_results.jsonl
   ```
   `merge` 给每个 `final.parquet` 加上 `geneval_correct` / `geneval_reason` 两列，重建 `final_all.parquet`，并按 `summary_scores.py` 的口径打印 GenEval 分数（6 类各自的准确率再取平均）。

## 输出

目录结构与设置文档一致（`prompts/0000/ … 0552/`，内含 `xT.npy`、`x0_latents.npy`、`xt_latents.npy`、`scores.parquet`、`final.parquet`、`emb_hps.npy`、`emb_blip.npy`、`final/seed_XX.png`、`DONE`），另有：

- `DONE`：这个 prompt 的完整配置、32 个 DDIM timestep、写它的 run_id 和耗时。`x0_latents.npy` 第 1 维的第 i 项对应 `config.x0_steps[i]` 步，`xt_latents.npy` 对应 `config.xt_steps[i]` 步。
- `run_meta/<run_id>.json`：每次启动一个文件，记录参数、GPU 型号、驱动、库版本、pip freeze、git commit、代码 md5、SD1.5 快照、耗时、显存峰值。设置文档里写的是一个 `run_meta.json`；这里有 4 个分片，还可能续跑，所以按次记录，`check_cache.py` 会核对这些记录是否一致。
- `prompts.jsonl`（含 `prompt_id` 和 GenEval 原始 metadata）、`folds.json`（按类别分层的 5 折，固定种子 0，各折 111/111/111/110/110 个 prompt）。
- 存哪些步可以用参数改：`--x0-steps all|scored`、`--xt-steps scored|all`、`--score-every`。

## 与设置文档的出入

1. **中期复现检查要换一种做法。** 缓存按设置文档用公共随机数：每个 prompt 都用种子 0–31，每个种子单独用一个 CPU `torch.Generator`。中期用的是 PSP 采集脚本的规则：seed = prompt_id × 16 + i，在 CUDA generator 上抽噪声。所以缓存里没有一条轨迹和中期对得上，"sub5 种子 0–15 复现中期 metrics.csv"这一项做不到。改为由 `verify_midterm.py` 把中期的噪声和 batch 大小喂给本代码，比较第 4、8、…、32 步和最终图的 IR。
2. **"第 32 步分数与 final 一致（差 < 1e-3）"不会成立，这是 DDIM 本身决定的，不是 bug。** SD1.5 的 DDIM 配置是 `set_alpha_to_one=False`，最终 latent 为 x_t⁽³²⁾ = √ᾱ₀·x̂₀⁽³²⁾ + √(1−ᾱ₀)·ε̂，ε̂ 的系数约 0.03。中期数据里 |IR(x̂₀⁽³²⁾) − IR(final)| 平均 0.025，只有 4% 小于 1e-3。改成两项检查：
   - 硬性：重新给 PNG 打分，与 `final.parquet` 的差 < 1e-3；
   - 只报告：第 32 步与 final 的差值分布。
3. **精度。** IR 沿用 PSP 自己的 ImageReward 封装（fp32），这样 IR 能和中期、PSP 的数字直接比较。HPS 和 hpsv2 一样用 autocast（fp16）。BLIP-ITM 用 fp16。UNet 和 VAE 是 fp16。
4. **HPS 只加载一次、成批计算。** `hpsv2.score` 每次调用都会重新读一遍 2 GB 的 checkpoint，PSP 又是逐张调用它，28 万张图用不了。这里沿用 hpsv2 的模型、预处理和 autocast，只是加载一次、按 batch 计算，同时取出 1024 维图像 embedding。
5. **BLIP-ITM 的前向自己写。** transformers 4.38.2 的 `BlipForImageTextRetrieval.forward` 把 ITM 的 cross-attention mask 建在 CPU 上，在 GPU 上会报错。这里按它的源码逐行写出来，把 mask 放到 GPU 上，同时让视觉塔只跑一次、ITM 和 ITC 共用。
6. **采样循环自己写。** PSP 的采集脚本在回调里为每个打分步多跑一次 UNet 来求 x̂₀；这里直接取 `scheduler.step` 返回的 `pred_original_sample`，不多跑 UNet。用小模型在 CPU 上核对过：最终 latent 与 PSP 管线逐位相同，每步 x̂₀ 与 PSP 采集脚本回调算出的逐位相同。在真实模型上由 `--verify-pipeline` 再核对一次。
7. **GenEval 只认 `\d+.png` 文件名。** 所以 `export` 用符号链接建一个 `samples/0000.png …` 视图，缓存里仍然是 `final/seed_XX.png`。

设置文档里的下游评估协议（回放 Standard / BoN / PSP / 随机剪枝 / 置信度剪枝、5 折交叉验证、bootstrap）属于缓存跑完之后的分析，不在这些脚本里。
