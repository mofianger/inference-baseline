# 新增模型、参考实现和测试集

## 交付物

1. `specs/models/<id>.json`：精确 checkpoint/revision 与资产哈希、模型配置、输入输出语义、默认步数/动作长度、允许变化范围。
2. `specs/protocols/<id>.json`：计时边界、输入设备、输出空间、工作量、同步、缓存规则和统计计划。
3. `specs/references/<id>.json`：参考实现 commit、环境/精度、噪声控制、输出定义。官方实现也要固定版本。
4. `specs/datasets/<id>.json`：数据来源、许可/访问方式、采样与预处理版本、样本角色、完整制品 manifest/hash。
5. 模型插件和至少一个真实 Adapter；完成一个可复核 Run 后标为 measured。

当前 JSON 是描述性目录，尚无自动解析器。可参照 GR00T 的四份声明，但不要把新 JSON 交给旧 CLI 后声称新模型已经支持。

## 输入约定

明确 batch、相机名称/顺序/时间对应、分辨率、颜色通道、裁剪/resize、状态维度/历史、坐标系、归一化、语言模板/tokenizer、mask、噪声算法和 dtype。三视角 256×256 是当前 workload，不是所有模型必须遵守的规则。

模型规格描述支持的输入语义；测试集固定本轮具体样本；协议选择本轮工作量。跨引擎共享 prepared tensor 的前提是语义兼容，而不是字典 key 看起来相同。

校准样本与评估样本分开，记录选择过程。数据变动、预处理或参考输出变动都生成新版本和新 hash。大型 tensor 留在制品存储/本地目录；Git 保存 manifest、获取方法、生成脚本及结果，不保存权重。

## 当前必须先改的代码

`ModelProtocol` 和 dataset builder 目前绑定 GR00T；Runner 的 action shape、state/noise 和输入响应 probe 也有模型假设。接入 π0.5 等新模型时，应先抽出模型插件的 build_inputs、validate_inputs、validate_outputs、reference 和 probes 职责，再复用 Runner 的预热/计时/统计。

不要为每个新模型复制一套 Runner；也不要只换模型名称就复用 GR00T 的 132 维、4 步或相机布局。协议要求模型完整输入到输出时，Adapter 必须补齐其原生入口前后遗漏的模型计算。

## 验证完成标准

固定样本能重复生成或按 hash 获取；同一噪声可明确传入参考和引擎；输出含义一致；不兼容的 dtype/shape/协议被拒绝；至少一次实测包含来源、误差、计时与限制。任务成功率需要单独的任务数据和评估协议，本仓库当前数值误差不替代任务评估。
