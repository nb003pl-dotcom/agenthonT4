# Final submission validation

The final source kit was built as a linux/amd64 Docker image and tested through the actual analysis verb. Synthetic House traffic ran only on an internal Docker network; no production model API or platform score was used. Tests cover successful submitted reasons and safe retention of the analysis answer when optional reasons fail. The original restored under25 directory remains unchanged.

## 已完成验证

- 52 项本地测试通过；包含恢复版门控/预算、预测批处理、事实归属、House 环境参数、完整实体发布、24+1 调用预算和实际 CLI 的理由成功/失败路径。
- 描述符另外使用官方 `qfbench2-common v2.6.1`（commit `99fd66565303554fbf9063140b45287e11895b8f`）复核，3 项测试通过；官方 `submission pack --help` 接口已确认。
- 镜像构建成功，所有锁定包安装成功，`pip check` 无依赖冲突。
- 镜像实际为 linux/amd64，接口标签为 2.0，无 VOLUME，入口接受 `analyze --task --corpus --out`。
- 内部网络实际容器测试全部通过：classification 4 次调用、regression 4 次、ranking（3 实体）6 次，均有合法 submitted_reasons；classification 的理由失败用例共 4 次调用，仍输出完整预测并明确记录 retained_analysis_answer。
- 容器测试输入只读挂载到 /input，输出挂载到 /output；设置平台式代理变量和 NO_PROXY，House 模型名、Bearer 与 /v1 路径均由模拟服务校验。
- 无网络容器内检查了 386 个应用运行文件，均与源码字节匹配；未打包题目目录、跑分结果、.env 或神经权重。
- 历史 workflow、prompts、profiles 保持原恢复版内容；调整集中在提交入口及 final 适配器。

源码包内 `validation/` 保存结构化验证、容器用例汇总、镜像元数据及应用文件哈希。工作区 `dist/container-smoke-verified/` 另有完整合成测试产物。

## 验证范围

未连接真正的 organizer House endpoint，未运行生产 NLI/reasoning judge，因此没有本提交包的官方质量分。没有推送公共仓库或上传 CodaBench。实际上传前仍需真实公开镜像 digest、队伍别名/编号、House 模型披露和用户选定的代码许可证，再由官方工具在本机使用 Team Key 签封。Team Key 不应进入代码、镜像或聊天。
