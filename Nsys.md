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

#hybrimoe运行 

```bash
CUDA_VISIBLE_DEVICES=2 python -m ktransformers.local_chat --model_path ./DeepSeek-V2-Lite-Chat-GGUF/config --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 16 --prefetch_size 0 --optimize_rule_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat.yaml
```



nsys已经下载好了，`version 2023.4.4.54-234433681190v0`

常用命令

```bash
nsys profile -o my_profile python my_script.py
```

存放路径

```bash
/data/home/tianjianyang/NSys_profiles
```

v0.1.4的老版ktrans

运行

```bash
CUDA_VISIBLE_DEVICES=2 nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) python -m ktransformers.local_chat --model_path ./DeepSeek-V2-Lite-Chat-GGUF/config --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF
```

注意，运行结束要等蛮久的（可能需要一个新的conda再install一遍）

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250318211721790.png" alt="image-20250318211721790" style="zoom: 80%;" />



hybrimoe老版

运行

```bash
nsys profile --duration=100 --stats=true python script.py

CUDA_VISIBLE_DEVICES=2 nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) python -m HybriMoE.local_chat --model_path ./DeepSeek-V2-Lite-Chat-GGUF/config --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 2 --optimize_rule_path HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat.yaml



CUDA_VISIBLE_DEVICES=2 nsys profile --duration=100 --stats=true -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) python -m HybriMoE.local_chat --model_path ./DeepSeek-V2-Lite-Chat-GGUF/config --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 8 --prefetch_size 2 --optimize_rule_path HybriMoE/optimize/optimize_rules/DeepSeek-V2-Chat.yaml
```

对话：请帮我写一本东方玄幻小说大纲

遇到时间戳报错可以再来一次，耐心等待

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250318215710842.png" alt="image-20250318215710842" style="zoom:80%;" />



HybriKtrans

运行

```bash
CUDA_VISIBLE_DEVICES=2 nsys profile -o /data/home/tianjianyang/NSys_profiles/NSys_$(date +%Y%m%d_%H%M%S) python -m ktransformers.local_chat --model_path ./DeepSeek-V2-Lite-Chat-GGUF/config --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 16 --prefetch_size 0 --optimize_rule_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat.yaml
```

耐心等待

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250318220456655.png" alt="image-20250318220456655" style="zoom:80%;" />



Ktransformers没有cuda_graph

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250320102641148.png" alt="image-20250320102641148" style="zoom:80%;" />



这里都把local_chat的while True注释掉了
