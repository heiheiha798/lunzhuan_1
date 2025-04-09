## debug

```cpp
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "HybriMoE",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "--model_path","/data/home/tianjianyang/HybriMoE/DeepSeek-V2-Lite-Chat-GGUF/config",
                "--gguf_path","/data/home/tianjianyang/HybriMoE/DeepSeek-V2-Lite-Chat-GGUF",
                "--cache_size","8",
                "--prefetch_size","2",
                "--optimize_rule_path","/data/home/tianjianyang/HybriMoE/HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat.yaml"

            ],
            "env":{
                "CUDA_VISIBLE_DEVICES":"2"
            },
        }
    ]
}

//ctrl+shift+p修改python path
```

#

```bash
#指定某个GPU
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
export CUDA_VISIBLE_DEVICES=0,1  # 使用第一个和第二个GPU

#查看GPU
echo $CUDA_VISIBLE_DEVICES
```



## 指导

要用hybrimoe，yaml文件中experts的device和op设置得是cuda和KExpertsMarlin



```bash
python HybriMoE/local_chat.py --model_path ./DeepSeek-V2-Lite --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 0 --optimize_config_path HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml

#prefetch_size不等于0貌似会报错
python HybriMoE/local_chat.py --model_path ./DeepSeek-V2-Lite --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 2 --optimize_config_path HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
#KEY ERROR

#Nsys
nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) \
--cuda-um-cpu-page-faults=false \
--cuda-um-gpu-page-faults=false \
--cuda-flush-interval=100 \
--cuda-memory-usage=true \
python HybriMoE/local_chat.py \
--model_path ./DeepSeek-V2-Lite \
--gguf_path ./DeepSeek-V2-Lite-Chat-GGUF \
--cache_size 8 \
--prefetch_size 0 \
--optimize_config_path HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml

nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) \
--cuda-um-cpu-page-faults=false \
--cuda-um-gpu-page-faults=false \
--cuda-flush-interval=100 \
--cuda-memory-usage=true \
python HybriMoE/local_chat.py \
--model_path ./DeepSeek-V2-Lite \
--gguf_path ./DeepSeek-V2-Lite-Chat-GGUF \
--cache_size 8 \
--prefetch_size 2 \
--optimize_config_path HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
```



注意

```python
print(f"loading {translated_key} to {device}")
```

在utils.py文件中，这个过程巨慢



## 报错

```bash
Traceback (most recent call last):
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/local_chat.py", line 158, in <module>
    local_chat(
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/local_chat.py", line 143, in local_chat
    generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/util/utils.py", line 222, in prefill_and_generate
    next_token = decode_one_tokens(
                 ^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/util/utils.py", line 117, in decode_one_tokens
    logits = model(
             ^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/models/modeling_deepseek.py", line 1588, in forward
    outputs = self.model(
              ^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/models/modeling_deepseek.py", line 1394, in forward
    layer_outputs = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/models/modeling_deepseek.py", line 1146, in forward
    hidden_states = self.mlp(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/operators/experts.py", line 1573, in forward
    self.moe_on_cpuinfer(hidden_states, topk_idx, topk_weight, shared_experts)
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/operators/experts.py", line 1604, in moe_on_cpuinfer
    outs = self.experts(x, topk_ids, topk_weight, shared_experts)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/anaconda3/envs/hybrimoe/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/operators/experts.py", line 1373, in forward
    return self.generate_experts.forward(input_tensor, expert_ids, weights, shared_experts)W
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/operators/experts.py", line 591, in forward
    self.cache.prefetch_expert(layer_idx=self.layer_idx)
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/operators/experts.py", line 1132, in prefetch_expert
    self.load_expert_weights(
  File "/data/home/tianjianyang/HybriMoE/HybriMoE/operators/experts.py", line 995, in load_expert_weights
    device = self.layer2device[layer_idx]
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^
KeyError: tensor(2, device='cuda:0')
```

这个是

```bash
python HybriMoE/local_chat.py \
--model_path ./DeepSeek-V2-Lite \
--gguf_path ./DeepSeek-V2-Lite-Chat-GGUF \
--cache_size 8 \
--prefetch_size 1 \
--optimize_rule_path HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
```

跑出来的，但是但从报错来看非常奇怪，为什么一个int会变成tensor？

代码有问题 line 1120

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250322221643071.png" alt="image-20250322221643071" style="zoom:50%;" />

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250322221704086.png" alt="image-20250322221704086" style="zoom:50%;" />

本来这里应该传一个int，但是变成了torch.Tensor

继续检查是expert_idx的问题。

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250322223718902.png" alt="image-20250322223718902" style="zoom:50%;" />

又因为experts是`get_experts_to_prefetch`返回的torch.long， 应该是一个张量，然后用`enumerate`依次提取

加一点输出，发现

```bash
experts:   		    tensor([ 3, 14], device='cuda:0')
idx:     			0
expert_idx:         tensor(3, device='cuda:0')
```

好的，缺了一点点东西，加上就好

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250322230746816.png" alt="image-20250322230746816" style="zoom:50%;" />

这又是什么东西？？？

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250322230805185.png" alt="image-20250322230805185" style="zoom:50%;" />

最后是有东西的。。。

跑一个nsys试试

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250322232326617.png" alt="image-20250322232326617" style="zoom:50%;" />

`prefetch_size=0`的overlap如下

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250323131144344.png" alt="image-20250323131144344" style="zoom:50%;" />

`prefetch_size=2`的overlap如下

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250323131656411.png" alt="image-20250323131656411" style="zoom:50%;" />







