# Under25 — AgentHon Track 4 submission code

This directory packages the current Under25 forecasting workflow for the AgentHon analysis interface. It uses only the organizer-injected House route and preserves the organizer's proxy environment. The final adapter reserves one request for validated submitted reasons while keeping the total at 25 requests. This source kit is not the sealed CodaBench upload: that upload must reference your publicly pullable image digest and include the official team claim.

## 使用方式

提交代码与原 `under25/` 分开保存，原代码和历史跑批均保持不变。生成模型只能由平台注入，入口不会读取 `.env`、OpenRouter key 或本地 provider 设置。

官方调用形式：

```bash
docker run --rm --platform linux/amd64 \
  -e MODEL_ENDPOINT -e MODEL_NAME -e MODEL_TOKEN \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
  -v /absolute/unit:/input:ro \
  -v /absolute/output:/output \
  under25-agenthon:final-20261027 \
  analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json
```

`MODEL_ENDPOINT` 必须是平台提供的 origin，程序追加 `/v1/chat/completions`。`MODEL_NAME` 原样使用，`MODEL_TOKEN` 只用于 Bearer 认证。平台代理必须保留；本地历史跑批的“取消代理直连”设置不应用到提交容器。

## 当前 query-plan 打包边界

- `workflow/` 已同步当前 Under25 实现，包含 task-first 6–8 needs、AC 词表索引扫描、表头 vocabulary、Post-earnings family gate，以及独立的 `task_only` planner 路线。
- `data/query-term-ac-index.json.gz` 随镜像离线提供；运行时不下载词表或索引。
- 官方入口保持 `query_plan_context=scanned` 的默认路径；`task_only` 作为实现能力随源码交付，但不会在固定提交入口中替换所有任务的默认路线。
- planner 仍受总请求预算约束；AC/表头扫描不消耗模型请求。
- 事实提取、预测批处理、按证据门控的内部 reason、定量与引用校验继续使用 10.25 逻辑。
- `submission_runtime.py` 将核心调用预算设为 24，再留 1 次生成最终 `submitted_reasons`；所有失败和重试仍计入总预算 25，单次输出上限 4000。
- 总时限为 task card 限额与 570 秒中的较小值；为最终理由最多预留 45 秒。该预留可能让核心在预算边界少做可选工作。
- 始终为 final；不支持通过环境变量切回 development。
- 最终理由解释已发布的预测，不能改预测值、标签或区间。理由必须通过官方确定性校验后才附加；格式、调用或时间失败时保留完整合法预测，不输出空理由数组或部分实体。
- `execution-report.json` 明确记录理由成功、跳过或保留分析答案的状态。
- 因为最终 House 模型由平台指定、预算也为理由留出了额度，不能把之前 OpenRouter development 的成绩当成本提交包的新成绩。

## 构建与验证

```bash
docker build --platform linux/amd64 -t under25-agenthon:final-20261027 .
```

镜像固定 Python 基础 digest，锁定全部依赖，带 `qfbench2.interface_version="2.0"` 标签，没有 `VOLUME`。Docker 构建只复制运行代码、提示词、版本配置、预构建 query-term AC 索引及必要离线分词/校验资源。不会复制题目 corpus、结果、参考答案、密钥、虚拟环境、工具测试或源码恢复记录。

本地测试：

```bash
python -m pip install -r requirements.txt pytest==9.1.1
python -m pytest -q
python tools/smoke_container.py --image under25-agenthon:final-20261027
```

容器 smoke 使用合成输入和模拟 House 服务，二者位于内部 Docker 网络；不调用外部模型、不花模型 API 额度。覆盖分类、回归、排名以及理由生成失败时保留预测。见 [VALIDATION.md](VALIDATION.md)。

## 生成真正的提交 ZIP

官方要求先公开推送镜像，再使用 registry digest，而不是本地 image ID。你需要提供自己的公开仓库、队伍编号、由 Team Key 派生的 team_id、实际 House 模型披露信息及你为自己代码选定的许可证。

安装官方固定工具版本：

```bash
python -m pip install -r tools/requirements-packaging.txt
```

先按官方 [image submission guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/IMAGE-SUBMISSIONS.md) 推送镜像并验证匿名拉取，取得 `registry/repository@sha256:...`。用官方命令生成队伍别名；Team Key 在本机隐藏输入，不要放到命令行、代码或聊天中：

```bash
qfbench2 submission alias --team-number "$TEAM_NUMBER"
```

将输出别名设置为 `TEAM_ID`，将公开镜像 digest 设置为 `IMAGE_DIGEST_REF`，将模型披露 JSON 文件路径设置为 `HOUSE_MODEL_DISCLOSURE`，并设置自己的 `SUBMISSION_LICENSE`。模型文件是官方要求的数组，填写实际模型的 name、version、training_cutoff、access、revision；access 必须为 api，不应填写示例或猜测的训练截止日期。

```bash
python tools/make_descriptor.py \
  --image "$IMAGE_DIGEST_REF" \
  --team-id "$TEAM_ID" \
  --models "$HOUSE_MODEL_DISCLOSURE" \
  --license "$SUBMISSION_LICENSE" \
  --out submission.json

qfbench2 submission pack \
  --descriptor submission.json \
  --team-number "$TEAM_NUMBER" \
  --out submission.zip
```

描述符从官方 `analysis_final` fixture 派生，并通过官方 `seal_descriptor_digest` 和 `SubmissionDescriptor.from_mapping` 校验；不自行重写签封算法。最终 pack 会再次在本机请求 Team Key，产出包含 `submission.json` 和 `team-claim.json` 的官方上传 ZIP。

## 来源和交付文件

[PROVENANCE.json](PROVENANCE.json) 记录恢复版来源与提交适配差异；[MANIFEST.sha256.json](MANIFEST.sha256.json) 记录本包文件字节。第三方资源许可证和 notices 保留在 `vendor/` 中；它们不替代你对自身提交代码选择的许可证。

`dist/under25-final-source.tar.gz` 为源码交付包，`dist/under25-final-image.tar.gz` 为可用 `docker load` 导入的本地镜像归档。二者都不是 CodaBench 的 `submission.zip`。

官方依据：[Track 4 CLI contract](https://github.com/Agenthon-2026/track4-analysis-public/blob/main/SUBMISSION_CLI.md)、[descriptor guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.6.1/starter-packs/track4/SUBMISSION-DESCRIPTOR.md)。
