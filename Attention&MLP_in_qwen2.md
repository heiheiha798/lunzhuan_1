# torch.nn

简介：Neural Network的简称

功能：创建、训练、保存、恢复神经网络

内容：包括nn.Parameter、nn.Linear、nn.functional、nn.Module、nn.Sequential

## nn.Linear

功能：对输入的 $x$ 做一个线性变换 $y=Wx+b$ ，用来创建一个多输入、多输出的全连接层——MLP显然要用

 代码：

```python
class torch.nn.Linear(in_features,out_features,bias=True)
```

举个例子

```python
# nn.Linear
# 建立单层的多输入、多输出全连接层
# in_features由输入张量的形状决定，out_features则决定了输出张量的形状 
full_connect_layer = nn.Linear(in_features = 28 * 28 * 1, out_features = 3)
print("full_connect_layer:", full_connect_layer)
print("parameters        :", full_connect_layer.parameters)
 
# 假定输入的图像形状为[64,64,3]
x_input = torch.randn(1, 28, 28, 1)
 
# 将四维张量转换为二维张量之后，才能作为全连接层的输入
x_input = x_input.view(1, 28 * 28 * 1)
print("x_input.shape:", x_input.shape)
 
# 调用全连接层
y_output = full_connect_layer(x_input) 
print("y_output.shape:", y_output.shape)
print("y_output:", y_output)

####
full_connect_layer: Linear(in_features=784, out_features=3, bias=True)
parameters        : <bound method Module.parameters of Linear(in_features=784,out_features=3, bias=True)>
x_input.shape: torch.Size([1, 784])
y_output.shape: torch.Size([1, 3])
y_output: tensor([[-0.2892, -0.3084,  0.9027]], grad_fn=<AddmmBackward>)
```

可以看到nn.Linear是没有之后的softmax/relu层的

## nn.functional

功能：包括神经网络前向和后向处理所需要到的常见函数，如神经元处理函数、各种激活函数等。

代码：

```python
# nn.functional.relu( )
print(y_output)
out = nn.functional.relu(y_output)
print(out.shape)
print(out)

####
tensor([[ 0.1023,  0.7831, -0.2368]], grad_fn=<AddmmBackward>)
torch.Size([1, 3])
tensor([[0.1023, 0.7831, 0.0000]], grad_fn=<ReluBackward0>)
```

```python
# nn.functional.sigmoid( )
print(y_output)
out = nn.functional.sigmoid(y_output)
print(out.shape)
print(out)

####
tensor([[ 0.1023,  0.7831, -0.2368]], grad_fn=<AddmmBackward>)
torch.Size([1, 3])
tensor([[0.5255, 0.6863, 0.4411]], grad_fn=<SigmoidBackward>)
```

## nn.Parameter

功能：是Tensor，也就是说是一个多维矩阵

代码：

```python
# nn.functional.linear( )
x_input = torch.Tensor([1., 1., 1.])
print("x_input.shape:", x_input.shape)
print("x_input      :", x_input)
print("")

####
x_input.shape: torch.Size([3])
x_input      : tensor([1., 1., 1.])
####

Weights1 = nn.Parameter(torch.rand(3))
print("Weights.shape:", Weights1.shape)
print("Weights      :", Weights1)
print("")

####
Weights.shape: torch.Size([3])
Weights      : Parameter containing:
tensor([0.3339, 0.7027, 0.9703], requires_grad=True)
####

Bias1 = nn.Parameter(torch.rand(1))
print("Bias.shape:", Bias1.shape)
print("Bias      :", Bias1)
print("")

####
Bias.shape: torch.Size([1])
Bias      : Parameter containing:
tensor([0.4936], requires_grad=True)
####

Weights2 = nn.Parameter(torch.Tensor(3))
print("Weights.shape:", Weights2.shape)
print("Weights      :", Weights2)

####
Weights.shape: torch.Size([3])
Weights      : Parameter containing:
tensor([0.0000e+00, 1.8980e+01, 1.1210e-44], requires_grad=True)
####

print("\nfull_connect_layer")
full_connect_layer = nn.functional.linear(x_input, Weights1)
print(full_connect_layer)

####
full_connect_layer
tensor(2.0068, grad_fn=<DotBackward>)
```

## nn.Module

抽象概念，可以表示NN中某layer，也可以表示一个多层NN

有点复杂，先不学

## nn.Sequential

功能：是一个有序的容器，该类将按照传入构造器的顺序，依次创建相应的函数，并记录在Sequential类对象的数据结构中，同时以神经网络模块为元素的有序字典也可以作为传入参数。

因此，Sequential可以看成是有多个函数运算对象，串联成的神经网络，其返回的是Module类型的神经网络对象。

代码：

```python
print("利用系统提供的神经网络模型类：Sequential,以参数列表的方式来实例化神经网络模型对象")
# A sequential container. Modules will be added to it in the order they are passed in the constructor. 
# Example of using Sequential
model_c = nn.Sequential(nn.Linear(28*28, 32), nn.ReLU(), nn.Linear(32, 10), nn.Softmax(dim=1))
print(model_c)
 
print("\n显示网络模型参数")
print(model_c.parameters)
 
print("\n定义神经网络样本输入")
x_input = torch.randn(2, 28, 28, 1)
print(x_input.shape)
 
print("\n使用神经网络进行预测")
y_pred = model.forward(x_input.view(x_input.size()[0],-1))
print(y_pred)

####
Sequential(
  (0): Linear(in_features=784, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=10, bias=True)
  (3): Softmax(dim=1)
)

显示网络模型参数
<bound method Module.parameters of Sequential(
  (0): Linear(in_features=784, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=10, bias=True)
  (3): Softmax(dim=1)
)>

定义神经网络样本输入
torch.Size([2, 28, 28, 1])

使用神经网络进行预测
tensor([[-0.1526,  0.0437, -0.1685,  0.0034, -0.0675,  0.0423,  0.2807,  0.0527,  -0.1710,  0.0668],
        [-0.1820,  0.0860,  0.0174,  0.0883,  0.2046, -0.1609,  0.0165, -0.2392,  -0.2348,  0.1697]], grad_fn=<AddmmBackward>)
```

## torch.rand && torch.randn

```python
import torch
# 生成一个形状为(3, 4)的张量，元素值在[0, 1)之间
tensor = torch.rand((3, 4))
print(tensor)

####
tensor([[0.1490, 0.7928, 0.0411, 0.5075],
        [0.0754, 0.8043, 0.7533, 0.1298],
        [0.5087, 0.1185, 0.6706, 0.9509]])
```

```python
import torch
# 生成一个形状为(3, 4)的张量，元素值服从标准正态分布
tensor = torch.randn((3, 4))
print(tensor)
print("均值:", tensor.mean().item())
print("标准差:", tensor.std().item())

####
tensor([[ 1.7490, -0.1910, -0.6926,  0.2398],
        [ 0.9135, -0.3359,  1.0442,  0.7824],
        [-0.7138,  0.1682, -1.2245, -0.3747]])
均值: 0.1137121245265007
标准差: 0.868906557559967
```



# Llama

## LlamaConfig

1. **`vocab_size` (`int`, *optional*, defaults to 32000)**:
   - 词汇表大小，定义了模型可以处理的不同 token 的数量
2. **`hidden_size` (`int`, *optional*, defaults to 4096)**:
   - 隐藏层表示的维度，即每个 token 的嵌入向量的维度。
3. **`intermediate_size` (`int`, *optional*, defaults to 11008)**:
   - MLP（多层感知机）层的中间维度
4. **`num_hidden_layers` (`int`, *optional*, defaults to 32)**:
   - Transformer 解码器中的隐藏层数量，就是模型层数
5. **`num_attention_heads` (`int`, *optional*, defaults to 32)**:
   - 每个注意力层中的注意力头数量。
6. **`num_key_value_heads` (`int`, *optional*)**:
   - 用于实现分组查询注意力（Grouped Query Attention, GQA）的键值头数量。如果未指定，默认与 `num_attention_heads` 相同。
   - 如果 `num_key_value_heads=num_attention_heads`，则使用多头注意力（MHA）。
   - 如果 `num_key_value_heads=1`，则使用多查询注意力（MQA）。
   - 否则，使用分组查询注意力（GQA）。
7. **`hidden_act` (`str` or `function`, *optional*, defaults to `"silu"`)**:
   - 解码器中使用的非线性激活函数，通常是 SiLU（Sigmoid Linear Unit）。
8. **`max_position_embeddings` (`int`, *optional*, defaults to 2048)**:
   - 模型支持的最大序列长度。LLaMA 1 支持最多 2048 个 token，LLaMA 2 支持最多 4096 个 token
9. **`initializer_range` (`float`, *optional*, defaults to 0.02)**:
   - 用于初始化所有权重矩阵的截断正态分布的标准差。
10. **`rms_norm_eps` (`float`, *optional*, defaults to 1e-06)**:
    - RMS 归一化层中使用的 epsilon 值，用于数值稳定性。
11. **`use_cache` (`bool`, *optional*, defaults to `True`)**:
    - 是否返回最后的键/值注意力（并非所有模型都使用）。仅在 `config.is_decoder=True` 时相关。
12. **`pad_token_id` (`int`, *optional*)**:
    - 填充 token 的 ID。
13. **`bos_token_id` (`int`, *optional*, defaults to 1)**:
    - 序列开始 token 的 ID。
14. **`eos_token_id` (`int`, *optional*, defaults to 2)**:
    - 序列结束 token 的 ID。
15. **`pretraining_tp` (`int`, *optional*, defaults to 1)**:
    - 预训练期间使用的张量并行度等级。这是一个实验性特性，用于确保预训练结果的精确复现。
16. **`tie_word_embeddings` (`bool`, *optional*, defaults to `False`)**:
    - 是否将输入和输出的词嵌入权重绑定在一起。
17. **`rope_theta` (`float`, *optional*, defaults to 10000.0)**:
    - RoPE（Rotary Position Embedding）嵌入的基周期。
18. **`rope_scaling` (`Dict`, *optional*)**:
    - 用于 RoPE 嵌入的缩放配置。可以指定不同的 RoPE 变体（如 'linear', 'dynamic', 'yarn', 'longrope', 'llama3'）以及相应的缩放因子。
19. **`attention_bias` (`bool`, *optional*, defaults to `False`)**:
    - 是否在自注意力层的查询、键、值和输出投影层中使用偏置。
20. **`attention_dropout` (`float`, *optional*, defaults to 0.0)**:
    - 注意力概率的 dropout 比率。
21. **`mlp_bias` (`bool`, *optional*, defaults to `False`)**:
    - 是否在 MLP 层的 `up_proj`, `down_proj` 和 `gate_proj` 中使用偏置。
22. **`head_dim` (`int`, *optional*)**:
    - 注意力头的维度。如果未指定，默认为 `hidden_size // num_attention_heads`。

注：

hidden_size是transformer中每个token转换后的维度，intermediate_size是前馈神经网络（FFN）中间层的维度。

RMS （Root Mean Square，方均根）归一化是一种层归一化的变体， $\hat{x_i}=\frac{x_i}{RMS(x)+\epsilon}$ ， $\epsilon$ 是防止分母为0的。

pad_token_id是用来补齐token长度的，bos_token_id和eos_token_id处理token，标记开始结束。

RoPE（Rotary Position Embedding，旋转编码），一种产生相对位置编码的方法，rope_theta是基周期。对于一个4维token，位置索引为10的每个维度旋转角度 $θ_{10,i}=\frac{10}{10000^{i/4}}$



## LlamaAttention

```python
"""Multi-headed attention from 'Attention Is All You Need' paper"""
```

这个就不往前看了

```python
class LlamaAttention(nn.Module):
	def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
```

自己定义了一个nn.Module类，然后扔了一堆自己的参数

其中的q、k、v、o都是用torch.nn.Linear定义的线性层，注意q输出维度是num_attention_heads * self.head_dim，是decoder前反馈输出的维度；kv都是num_key_value_heads * self.head_dim，有可能是GQA；o是输出的线性层，用来把自注意力输出转换为token维度。

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

然后是这个Forward函数，定义了具体的计算。

kwargs 是 keyword arguments 的缩写，表示关键字参数；**表示关键字参数， 它本质上是一个 dict，Llama这里是留下了FlashAttention优化的参数空间

query_states定义q空间的向量，将输入的hidden_states通过q_proj映射到q空间，view用于改变张量的形状成hidden_shape，然后transpose交换第1和第2维度，为了在后续的矩阵乘法中方便计算。

这里又有一个新函数apply_rotary_pos_emb，

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

根据他的注释，其实就是实现了RoPE。

接下来看到对cache的应用，也就是kvcache。

```python
if past_key_value is not None:
	cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
	key_states, value_states = past_key_value.update(key_states, value_states,self.layer_idx, cache_kwargs)
```

这里具体的update还没看。

接下来是一个选择attention机制的代码，

```python
attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```

这里去到具体的位置，可以看到有三种attention方法

```python
ALL_ATTENTION_FUNCTIONS.update(
    {
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
    }
)
```

这里放最后学，最后是

```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

把attn_output 的形状调整为与输入张量input_shape相同，用o_proj处理后变成hidden_size的向量，最后输出attn_output 和 attn_weights。

## 三种Attention的具体实现

### flash_attention_2

核心是剔除非矩阵运算，加速GPU效率，同时引入了矩阵S储存$QK^T$的结果，并采用分块计算和增量更新的方式，使得并行度提高。具体操作是将QKV矩阵分块，计算$QK^T$​储存在S，增量计算softmax，流式更新结果矩阵O

```python
if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func
```

来自FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness，不想学了。



### flex_attention_forward（torch.nn.attention内置的，提供了一个灵活的 API，可以实现多种 Attention 变体）

经典的注意力评分：

```python
score: Tensor[batch_size, num_heads, sequence_length, sequence_length] = (Q @ K) / sqrt(head_dim)
```

flex一点：

```python
Tensor[batch_size, num_heads, sequence_length, sequence_length] = score_mod(score)
```

举个例子：Relative Position Encodings

```python
def relative_positional(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)
```

Llama的flex_attention.py包括

```python
__all__ = [
    "BlockMask",
    "flex_attention",
    "create_block_mask",
    "create_mask",
    "create_nested_block_mask",
    "or_masks",
    "and_masks",
    "noop_mask",
]
```

真的都要学吗。。。。。。。#todo



### sdpa_attention_forward（torch.nn.functional内置的，最基础的单头注意力机制）

```python
#声明和初始化
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
```

没什么可说的

```python
#因果注意力屏蔽矩阵上三角
	if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
```

如果启用因果注意力（`is_causal=True`），则生成一个下三角矩阵 `temp_mask`，其中下三角部分为 `True`，上三角部分为 `False`。

将 `attn_bias` 中对应 `temp_mask` 为 `False` 的位置填充为负无穷（`-inf`），以屏蔽未来位置的信息。

将 `attn_bias` 转换为与 `query` 相同的数据类型

```python
#掩码部分设置为-inf
if attn_mask is not None:
    if attn_mask.dtype == torch.bool:
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
        attn_bias = attn_mask + attn_bias
```

如果提供了 `attn_mask`：

如果 `attn_mask` 是布尔类型，则将 `attn_bias` 中对应 `attn_mask` 为 `False` 的位置填充为负无穷（`-inf`）。

如果 `attn_mask` 是数值类型，则将其与 `attn_bias` 相加。

```python
#Grouped Query Attention 
if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
```

GQA设置KV维度和Q维度相同

```python
#数据处理
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
```

经典$A=Softmax(\frac{QK^T}{\sqrt{d_k}})$​ ，加了bias和dropout，最后weight和value矩阵相乘得到结果。



## LlamaMLP

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

初始化就不说了，gate_proj是把token维度从hidden_size升到intermediate_size的门控线性层，up_proj是升维，down_proj是降维，act_fn是"silu"。

重要的是forward函数，先将输入x过门控gate_proj，silu一下得到权重，乘到up_proj(x)上，就是MLP隐藏层中做了一个激活函数，然后down_proj再输出。这里比最基础的transformer多了一个gate_proj，貌似能提高性能。



# qwen2

发现

```python
class Qwen2MLP(LlamaMLP):
    '''
    '''
class Qwen2Attention(LlamaAttention):
    '''
    '''
```

所以就学一下不同的地方

```python
sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
```

就多了一个Sliding Window Attention，每个 token 仅关注局部窗口内的其他 token，而不是整个序列。
