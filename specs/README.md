# 研究契约目录

这些 JSON 为版本化的描述性契约，`document_only=true` 表示当前 CLI 不直接加载它们。实际执行入口仍是 `configs/` 与 dataset manifest，迁移解析器之前需人工核对两者一致。

- [模型](models/gr00t-n17-3b-base-v1.json)
- [协议](protocols/groot-n17-b1-v3-model-only-v1.json)
- [数据集](datasets/groot-n17-synthetic-v3.json)
- [参考实现](references/groot-n17-official-bf16-v1.json)

ID 稳定且唯一，语义改变增加版本；权重、数据和参考输出改变不得复用原 ID/hash。机器绝对路径属于执行配置，不属于可移植模型身份。
