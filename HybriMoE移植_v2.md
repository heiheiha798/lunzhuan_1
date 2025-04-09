#ladder

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export {HTTP,HTTPS,FTP,RSYNC}_PROXY=$http_proxy
```

#Adding CUDA to PATH

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda
#检查
echo $CUDA_HOME
echo $PATH
echo $LD_LIBRARY_PATH
```



## git配置

```bash
git config --global user.email "2300012738@stu.pku.edu.cn"
git config --global user.name "heiheiha798"

#查看git全局配置
git config --global --list
```



## 关于仓库管理

```bash
#创建一个名为 `ytj` 的分支：
git checkout -b ytj

#推送branch
git push origin ytj
git checkout -b ytj

#从ytj的branch拉去代码
git checkout ytj
git pull origin ytj

#切换branch以及查看
git checkout ytj-test
git branch

#修改完成后查看修改
git status  
#提交文件 git add .是全部文件
git add src/main.py 
#添加注释
git commit -m "更新了 main.py 的功能" 
git push origin ytj-test

#合并分支
git checkout ytj     
# 将 ytj-test 合并到 ytj
git merge ytj-test
```



## 代码移植

### install

```bash
conda deactivate 
conda create --name hybriktrans python=3.11
conda activate hybriktrans
conda install -c conda-forge libstdcxx-ng
strings ~/anaconda3/envs/hybriktrans/lib/libstdc++.so.6 | grep GLIBCXX
pip install torch torchvision torchaudio packaging ninja cpufeature numpy
cd HybriMoE-Release
git submodule init
git submodule update #这里third-party超级容易出问题，build失败十有八九是，往下面看
pip install flash_attn --verbose
bash install.sh

#
git clone --branch ytj https://github.com/shuzhangzhong/HybriMoE-Release.git
```

试运行

```bash
CUDA_VISIBLE_DEVICES=3 python -m ktransformers.local_chat --model_path /data/home/tianjianyang/DeepSeek-V2-Lite --gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF
```

没有问题

但是，当输入

```
请详细阐述广义相对论中的时空弯曲概念,并结合爱因斯坦场方程解释质量如何影响时空几何结构,进一步分析黑洞的形成机制及其边界事件视界的物理意义,同时探讨引力波的存在及其在LIGO实验中的探测原理,最后讨论宇宙学中的暗物质与暗能量问题,解释它们如何通过引力效应影响宇宙大尺度结构的演化,并分析当前宇宙加速膨胀现象与爱因斯坦最初引入的宇宙学常数之间的关系,以及现代观测数据对标准宇宙学模型的验证与挑战
```

会报错，发现是全角符号的问题，现在改成半角，`local_chat.py`中的`content`就设置成这个。

接下来的运行指令

```bash
python ktransformers/local_chat.py --model_path /data/home/tianjianyang/DeepSeek-V2-Lite --gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 0 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml

#prefetch_size不等于0会报错，做了一个简短的debug，改了一个tensor，不知是否有其他问题
python ktransformers/local_chat.py --model_path /data/home/tianjianyang/DeepSeek-V2-Lite --gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 2 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml

#Nsys-这里可以考虑只追踪cuda
nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) \
--cuda-um-cpu-page-faults=false \
--cuda-um-gpu-page-faults=false \
--cuda-flush-interval=100 \
--cuda-memory-usage=true \
python ktransformers/local_chat.py \
--model_path /data/home/tianjianyang/DeepSeek-V2-Lite \
--gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF \
--cache_size 8 \
--prefetch_size 0 \
--optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml

nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) \
--cuda-um-cpu-page-faults=false \
--cuda-um-gpu-page-faults=false \
--cuda-flush-interval=100 \
--cuda-memory-usage=true \
python ktransformers/local_chat.py \
--model_path /data/home/tianjianyang/DeepSeek-V2-Lite \
--gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF \
--cache_size 8 \
--prefetch_size 2 \
--optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml

nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) \
--trace=cuda \
--cuda-flush-interval=100 \
--cuda-memory-usage=true \
python ktransformers/local_chat.py \
--model_path /data/home/tianjianyang/DeepSeek-V2-Lite \
--gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF \
--cache_size 8 \
--prefetch_size 2 \
--optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
```

对比主要用到`https://www.json.cn/diff/` 。



### 1.experts.py

#### Marlin

line 418

load_gguf_tensor改成了get_mmap_tensor（？为什么）

```
https://github.com/kvcache-ai/ktransformers/commit/c189d55bd1d0b585933653420e90dbf4a25e2743#diff-9a5f493d9be19a6579339fc678c0b49c20a1a8654f3ad03886222400d282fc2a
```

line 437

forward 改了传参，用到Marlin的都要注意是否要修改

很多地方改成了

```bash
		if w is None:
            w = self.load_weights()
            load_by_experts = True
```

不知道逻辑是否有变化

#### Torch

多了一个

```python
    def load_weights(self, override_key: str | None = None):
```

之前也有一个`get_mmap_tensor`，不知道为什么改的

`forward`中也有不少dtype的修改，暂且和ktrans保持一致

moe_on_cpuinfer改成了moe_kexperts



#### optimize.py

#### linear.py

#### modeling_Deepseek.py

#### Experts

#### moe.h和moe.cpp

cpp和h对不上？？？

```C++
void forward_many(int qlen, int k, const uint64_t* size_per_token,const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend)

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const uint64_t* size_per_token, const float* weights, const void* input, void* output, Backend* backend) 
```

#### ext_binding.cpp



注：这里的修改细节和HybriMoE_to_Ktransformers文件有些许类似



#### third_patry中llama.cpp和pybind11库可能都有问题

根据ktransformers中的内容，尝试将branch切换为老版

修改完之后再次

```bash
bash install.sh
```

成功（不成功看terminal中哪里开始fail慢慢debug）



运行测试

```bash
python ktransformers/local_chat.py --model_path /data/home/tianjianyang/DeepSeek-V2-Lite --gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 0 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
```



报错，关于experts.py中KExpertsMarlin的line 366（大致这个位置）的move函数，做了一些修改，不知是否会出新bug。



跑通了，有点小警告，无伤大雅

```bash
/data/home/tianjianyang/HybriMoE-Release/ktransformers/operators/experts.py:712: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  top_k_scores, top_k_id = torch.topk(torch.tensor(priority_scores), k)
```



再试一下prefetch

```bash
python ktransformers/local_chat.py --model_path /data/home/tianjianyang/DeepSeek-V2-Lite --gguf_path /data/home/tianjianyang/DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 2 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
```

也没有问题



可能有问题

```bash
loading blk.23.ffn_gate_exps.weight with CPU
loading blk.23.ffn_up_exps.weight with CPU
loading blk.23.ffn_down_exps.weight with CPU
```

hybrimoe这里全是cuda:0 ，同时GPU利用率显著高，还需debug

```bash
grep -rnw '/data/home/tianjianyang/HybriMoE-Release' -e 'blk'
```

/data/home/tianjianyang/HybriMoE-Release/ktransformers/util/custom_gguf.py中`load_gguf_tensor`的问题，新版多了一个print，貌似没有问题
