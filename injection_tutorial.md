# 教程——Github英文翻译

## 摘要
本教程将指导您如何使用KTransformers框架将自定义操作符注入到模型中。我们将以DeepSeekV2-Chat模型为例，逐步演示如何将自定义操作符注入到模型中。教程将涵盖以下主题：
* [如何编写注入规则](#如何编写注入规则)
    * [理解模型结构](#理解模型结构)
* [多GPU支持](#多GPU支持)    
* [如何编写新操作符并将其注入到模型中](#如何编写新操作符并将其注入到模型中)

## 如何编写注入规则
Inject框架的注入规则基本形式如下：
```yaml
- match:
    name: "^model\\.layers\\..*\\.*$"  # 目标模块名称
    class: torch.nn.Linear  # 目标模块
  replace:
    class: "default"
    kwargs:
      generate_device: "cuda:0"
      # your_op_param_1: 1234
      # your_op_param_2: 5678
  recursive: True
```
* match: 该字段标记匹配规则，可以以两种形式出现，name和class。这两种匹配规则可以同时出现或单独出现；只有当两个条件都满足时才会匹配。
* replace:
	* class: 可以导入的Python类，用于替换目标模块。如果不需要替换，设置为default。
	* kwargs: 模块初始化所需的参数列表。
	    * generate_device: 该模块的设备，可以设置为“cpu”、“cuda”、“cuda:1”等。
* recursive: 是否递归注入该模块的子模块，默认为True。

对于recursive字段：某些模块包含多个子模块，例如Self-attention模块通常包括q/k/v/o四个线性模块。如果我们替换了self-attention模块，但不希望内部的线性模块被其他规则覆盖，请将此规则设置为False。

## 理解模型结构
以[deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)为例，我们可以按照上述规则逐步注入我们的自定义模块并运行。KTransformers提供了高度的灵活性，允许您替换/实验基本操作符。然而，这也要求用户清楚地了解他们正在运行的模型的结构。

幸运的是，了解模型的结构非常简单。打开[deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/main)主页上的文件列表，您可以看到以下文件：

<img src="https://github.com/kvcache-ai/ktransformers/raw/main/doc/assets/model_structure_guild.png" alt="Inject-Struction" style="zoom:50%;" />

从`.saftensors`文件中，我们可以看到每一层权重的名称，对应于注入规则中的match.name属性。
从`modeling_deepseek.py`文件中，我们可以看到每个模块类的具体实现，类名对应于注入规则中的match.class属性。

DeepSeekV2模型的结构如下：

<img src="https://github.com/kvcache-ai/ktransformers/raw/main/doc/assets/deepseekv2_structure.png" alt="Inject-Struction" style="zoom:50%;" />


支持的操作符及其对应的类如下：

| match     | replace                | backends                | descriptions                                |
| --------- | ---------------------- | ----------------------- | ------------------------------------------- |
| Linear    | KTransformersLinear    | KLinearMarlin           | Marlin作为后端                              |
|           |                        | KLinearTorch            | pytorch作为后端                             |
|           |                        | KLinearCPUInfer         | llamafile作为后端                           |
|           |                        | KLinearFP8              | Triton fp8_gemm内核。要求GPU能够计算fp8数据 |
| experts   | KTransformersExperts   | KExpertsTorch           | pytorch作为后端                             |
|           |                        | KExpertsMarlin          | Marlin作为后端                              |
|           |                        | KExpertsCPU             | llamafile作为后端                           |
| Attention | KDeepseekV2Attention   | KDeepseekV2Attention    | MLA实现                                     |
| MoE       | KMistralSparseMoEBlock | KQwen2MoeSparseMoeBlock | Qwen2的MoE                                  |
|           | KDeepseekV2MoE         | KDeepseekV2MoE          | DeepseekV2的MoE                             |
| Model     | KQwen2MoeModel         | KQwen2MoeModel          | Qwen2的模型                                 |
|           | KDeepseekV2Model       | KDeepseekV2Model        | DeepseekV2的模型                            |
| RoPE      | RotaryEmbedding        | RotaryEmbedding         | RoPE模块                                    |
|           | YarnRotaryEmbedding    | YarnRotaryEmbedding     | RoPE模块                                    |

然后我们开始逐步注入自定义模块，我们的目标是：

* Replace the linear module with custom Marlin linear module.
* Replace the self-attention module with a custom Absorption-based MLA module.
* Replace the experts module with a custom Experts module.
* Replace the MoE module with a custom MoE module.
* Replace the RoPE module with a custom RoPE module.
* 为每个模块设置运行设备。

完整的注入规则实现可以在[这里](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat.yaml)找到。

## Matrix Absorption-based MLA 注入

对于Attention模块的注入，我们只需要使用正则表达式匹配transformers中使用的模块名称，并将其替换为我们自己的MLA模块实现。YAML注入规则如下：
```yaml
- match:
    name: "^model\\.layers\\..*\\.self_attn$"  # 正则表达式
  replace:
    class: ktransformers.operators.attention.KDeepseekV2Attention # 优化的MLA实现
```
如您所见，YAML文件中的每个规则都有两部分：match和replace。match部分指定要替换的模块，replace部分指定要注入到模型中的模块以及初始化关键字。

## Routed Experts的注入
对于路由Experts（对应图中的exps），我们注入的模块是CPUInfer，它被包装在KTransformersExperts包装模块中。KTransformersExperts有多个实现，我们需要指定关键字来告诉包装模块我们要使用哪个实现以及我们计划如何使用它。

在transformer的源代码中，MoE是使用nn.ModuleList实现的。我们不希望KTransformers遍历列表中的所有子模块并逐个注入它们，因此在此规则中，我们设置recursive: False以防止递归注入到此模块的子模块中。YAML规则如下：

```yaml
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # 自定义MoE内核，支持专家并行
    kwargs:
      generate_device: "cpu"
      generate_op: "MLPCPUExperts"
      out_device: "cuda"
  recursive: False # 不要递归注入此模块的子模块
```

如果我们注入路由Experts作为自定义模块，则无法使用原始`nn.ModuleList`中的接口。因此，有必要修改FFN模块中的forward函数。最简单的方法是实现一个具有自定义forward函数的新模块并注入它。
```yaml
- match:
    class: ktransformers.models.modeling_deepseek.DeepseekV2MoE
  replace:
    class: ktransformers.operators.experts.KDeepseekV2MoE     # 具有自定义forward函数的MLP模块
```

## Linear Layers的注入

对于剩余的线性层模块，我们旨在使用量化操作符来节省存储空间，同时提高性能。由于目前没有关于MLA和量化结合的研究，我们不希望将线性注入到MLA操作符中。因此，我们可以修改正则表达式并在规则的match部分添加类型检查。只有同时匹配名称和类的模块才会被注入。我们还需要传递一些类似于路由Experts注入的关键字。YAML规则如下：

```yaml
- match:
    name: "^model\\.layers\\.(?!.*self_attn).*$"  # 正则表达式
    class: torch.nn.Linear  # 仅匹配同时匹配名称和类的模块
  replace:
    class: ktransformers.operators.linear.KTransformersLinear  # 在量化数据类型上优化的内核
    kwargs:
      generate_device: "cuda"
      generate_op: "QuantizedLinearMarlin"
```
## 预计算缓冲区的模块注入

为了避免在初始化注入的原始模型时占用资源，我们使用torch的meta设备来初始化原始模型。RoPE模块在初始化期间预计算了一些缓冲区，但在使用meta设备时不执行任何计算。因此，我们需要在加载模型时补偿缓冲区的计算。简单地说，我们将一个自定义模块注入到rotary embedding模块中，该模块在加载期间执行预计算。YAML规则如下：
```yaml
- match:
    class: ktransformers.models.modeling_deepseek.DeepseekV2YarnRotaryEmbedding
  replace:
    class: ktransformers.operators.RoPE.YarnRotaryEmbedding
```

## 为模块指定运行设备

最后，我们为所有模块设置一个后备基本属性generate_device：
```yaml
- match:
    name: "^model\\.layers\\..*\\.|^lm_head"
  replace:
    class: "default"
    kwargs:
      generate_device: "cuda"
  
- match:
    name: "^model.embed_tokens"
  replace:
    class: "default"
    kwargs:
        generate_device: "cpu"
```
通过这两个规则，我们将所有先前未匹配的层（及其子模块）和lm_head放在cuda上，将embedding放在cpu上。请注意，模块的属性将由它匹配的第一个规则决定。例如，如果您稍后在注入的模块中设置了新的replace.kwargs.generate_device，则较早设置的设备将优先。如果您的计算机有多个卡，您还可以将模型配置到多个卡上。


## 多GPU支持

如果您有多个GPU，您可以将每个模块的设备设置为不同的GPU。 
DeepseekV2-Chat有60层，如果我们有2个GPU，我们可以将30层分配给每个GPU。完整的多GPU规则示例[在这里](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-multi-gpu.yaml)。

<img src="https://github.com/kvcache-ai/ktransformers/raw/main/doc/assets/multi_gpu.png" alt="Inject-Struction" style="zoom: 50%;" />


首先，对于多GPU，我们必须注入一个新的操作符`KDeepseekV2Model`。并将层分配到不同的GPU。对于我们的情况，我们必须设置`KDeepseekV2Model`操作符中的`transfer_map`如下：

```yaml
- match:
    name: "^model$"
  replace:
    class: "ktransformers.operators.models.KDeepseekV2Model"
    kwargs:
      transfer_map: 
        30: "cuda:1"
```

我们还必须为模型中的每个模块设置设备。 

例如，对于`routed experts`，单个GPU的yaml是：
```yaml
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # 自定义MoE内核，支持专家并行
    kwargs:
      generate_device: "cuda:0"
      generate_op: "MLPCUDAExperts"
      out_device: "cuda:0"
  recursive: False # 不要递归注入此模块的子模块
```
但对于两个GPU，我们需要为模型中的每个模块设置设备。 

```yaml
# 将0-29层的out_device分配到cuda:0
- match:
    name: "^model\\.layers\\.(0|[1-9]|[12][0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # 自定义MoE内核，支持专家并行
    kwargs:
      generate_device: "cpu"
      generate_op:  "KExpertsCPU"
      out_device: "cuda:0"
  recursive: False # 不要递归注入此模块的子模块

# 将30-59层的out_device分配到cuda:1
- match:
    name: "^model\\.layers\\.([345][0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # 自定义MoE内核，支持专家并行
    kwargs:
      generate_device: "cpu"
      generate_op:  "KExpertsCPU"
      out_device: "cuda:1"
  recursive: False # 不要递归注入此模块的子模块
```
对于其他模块，我们可以以相同的方式设置设备。

## 如何编写新操作符并将其注入到模型中

在本节中，我们将解释如何编写一个可以注入的操作符，以新的线性实现为例。

首先，所有可注入的操作符都需要继承自BaseInjectedModule类，该类继承了我们注入框架所需的一些属性。它的初始化函数需要满足以下基本格式：

```python
class LinearTorchInject(BaseInjectedModule):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, generate_device, **kwargs)
```
如果用户有其他需要传递给此类的参数，也可以包含在init函数中，并在yaml文件中的kwargs参数中重新传递。例如，如果我们的操作符想要传递一个参数`my_param`，init函数可以写成：
```python
class LinearTorchInject(BaseInjectedModule):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "cuda",
        my_param: bool = True,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.my_param = my_param
```
然后我们的注入规则可以写成：
```yaml
- match: 
    name: "^model\\.layers\\..*$"  # 正则表达式匹配模块名称。
    class: torch.nn.Linear  # 可以添加类型限制。
  replace:
    class: ktransformers.operators.linear.LinearTorchInject  # 注入模块路径
    kwargs: # 额外参数
      generate_device: "cuda"
      my_param: True
```
对于线性模块，还需要从gguf文件中读取权重。我们提供了`KLinearBase`类来帮助用户从gguf文件中读取权重。用户只需要继承并实现load、unload和forward函数。因此，一个完全可注入的线性类将如下所示：
```python
class LinearTorchInject(BaseInjectedModule, KLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, generate_device, **kwargs)
        KLinearBase.__init__(self)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.w = None
        self.has_bias = False
    
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: w = self.load_weight(device=device)

        if isinstance(w, nn.Parameter):
            self.w = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T
            self.has_bias = False
        elif isinstance(w, tuple):
            self.w = w[0].to(dtype=self.dtype).view(self.out_features, self.in_features).T
            self.bias = w[1].to(dtype=self.dtype)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        self.w = self.w.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)

    def unload(self):
        if self.w is not None:
            self.w = None
        if self.has_bias:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out_device = x.device
        x = x.to(device=self.device, dtype=self.dtype)
        x = x @ self.w
        if self.has_bias:
            x = x + self.bias
        x = x.to(dtype=dtype, device=out_device)
        return x
```
请注意，`self.load_weight`函数由KLinearBase类提供，用于帮助用户将权重从gguf文件加载到模块中。KLinearBase的实现细节可以在[GITHUB](https://github.com/kvcache-ai/ktransformers/blob/44f57270c9514d79fab224186d90ccf61059331a/ktransformers/operators/linear.py#L31)上找到。
