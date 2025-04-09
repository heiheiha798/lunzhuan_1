Nsys (NVIDIA Nsight Systems) 是一个强大的性能分析工具，用于分析和优化应用程序的性能。`nsys profile` 命令用于启动性能分析会话，并提供了多种选项来定制分析过程。以下是一些常用的 `nsys profile` 选项及其用法：

## 基本用法
```bash
nsys profile [<args>] [application] [<application args>]
```
- `application` 是你想要分析的应用程序。
- `<application args>` 是传递给应用程序的参数。

## 常用选项

1. **`-b, --backtrace=`**
   - 指定用于采样的回溯方法。
   - 可选值：`lbr`, `fp`, `dwarf`, `none`。
   - 默认值：`lbr`。
   - 示例：`nsys profile -b dwarf ./my_app`

2. **`-c, --capture-range=`**
   - 指定何时开始分析。
   - 可选值：`none`, `cudaProfilerApi`, `nvtx`, `hotkey`。
   - 默认值：`none`。
   - 示例：`nsys profile -c cudaProfilerApi ./my_app`

3. **`--capture-range-end=`**
   - 指定捕获范围结束时的行为。
   - 可选值：`none`, `stop`, `stop-shutdown`, `repeat[:N]`, `repeat-shutdown:N`。
   - 默认值：`stop-shutdown`。
   - 示例：`nsys profile --capture-range-end=repeat:3 ./my_app`

4. **`-d, --duration=`**
   - 指定分析的持续时间（秒）。
   - 默认值：`0`（无限）。
   - 示例：`nsys profile -d 10 ./my_app`

5. **`-e, --env-var=`**
   - 设置应用程序进程的环境变量。
   - 示例：`nsys profile -e "CUDA_VISIBLE_DEVICES=0" ./my_app`

6. **`-o, --output=`**
   - 指定输出报告文件名。
   - 默认值：`report%n`。
   - 示例：`nsys profile -o my_report ./my_app`

7. **`-s, --sample=`**
   - 指定 CPU IP/回溯采样的范围。
   - 可选值：`process-tree`, `system-wide`, `none`。
   - 默认值：`process-tree`。
   - 示例：`nsys profile -s system-wide ./my_app`

8. **`-t, --trace=`**
   - 指定要跟踪的 API。
   - 可选值：`cuda`, `nvtx`, `cublas`, `mpi`, `opengl`, 等。
   - 默认值：`cuda,nvtx,osrt,opengl`。
   - 示例：`nsys profile -t cuda,mpi ./my_app`

9. **`--stats=`**
   - 生成摘要统计信息。
   - 可选值：`true`, `false`。
   - 默认值：`false`。
   - 示例：`nsys profile --stats=true ./my_app`

10. **`-w, --show-output=`**
    - 控制是否显示目标进程的标准输出和标准错误。
    - 可选值：`true`, `false`。
    - 默认值：`true`。
    - 示例：`nsys profile -w false ./my_app`

## 高级选项

1. **`--cuda-flush-interval=`**
   - 设置 CUDA 数据缓冲区的自动保存间隔（毫秒）。
   - 默认值：`0`。
   - 示例：`nsys profile --cuda-flush-interval=100 ./my_app`

2. **`--cuda-memory-usage=`**
   - 跟踪 GPU 内存使用情况。
   - 可选值：`true`, `false`。
   - 默认值：`false`。
   - 示例：`nsys profile --cuda-memory-usage=true ./my_app`

3. **`--cuda-um-cpu-page-faults=`**
   - 跟踪统一内存中的 CPU 页错误。
   - 可选值：`true`, `false`。
   - 默认值：`false`。
   - 示例：`nsys profile --cuda-um-cpu-page-faults=true ./my_app`

4. **`--cuda-um-gpu-page-faults=`**
   - 跟踪统一内存中的 GPU 页错误。
   - 可选值：`true`, `false`。
   - 默认值：`false`。
   - 示例：`nsys profile --cuda-um-gpu-page-faults=true ./my_app`

5. **`--event-sample=`**
   - 启用事件采样。
   - 可选值：`system-wide`, `none`。
   - 默认值：`none`。
   - 示例：`nsys profile --event-sample=system-wide ./my_app`

6. **`--gpu-metrics-device=`**
   - 从指定设备收集 GPU 指标。
   - 默认值：`none`。
   - 示例：`nsys profile --gpu-metrics-device=0 ./my_app`

7. **`--gpu-metrics-frequency=`**
   - 指定 GPU 指标采样频率。
   - 默认值：`10000`。
   - 示例：`nsys profile --gpu-metrics-frequency=5000 ./my_app`

8. **`--gpu-metrics-set=`**
   - 指定 GPU 指标采样集。
   - 示例：`nsys profile --gpu-metrics-set=1 ./my_app`

## 其他选项

1. **`--resolve-symbols=`**
   - 解析捕获的样本和回溯的符号。
   - 可选值：`true`, `false`。
   - 默认值：`true`（Windows 上为 `false`）。
   - 示例：`nsys profile --resolve-symbols=true ./my_app`

2. **`--run-as=`**
   - 以指定用户身份运行目标应用程序。
   - 示例：`nsys profile --run-as=myuser ./my_app`

3. **`--session-new=`**
   - 在新命名会话中启动收集。
   - 示例：`nsys profile --session-new=my_session ./my_app`

4. **`--start-frame-index=`**
   - 在达到指定帧索引时开始记录会话。
   - 示例：`nsys profile --start-frame-index=100 ./my_app`

5. **`--stop-on-exit=`**
   - 在启动的应用程序退出时停止分析。
   - 可选值：`true`, `false`。
   - 默认值：`true`。
   - 示例：`nsys profile --stop-on-exit=false ./my_app`

## 总结
`nsys profile` 提供了丰富的选项来定制性能分析过程。通过合理使用这些选项，你可以更精确地控制分析的范围、深度和输出格式，从而更有效地优化应用程序的性能。
