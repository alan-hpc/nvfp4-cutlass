# BF16 × NVFP4 Dual-A GEMM 实现纪要

本文记录 `include/nvfp4_gemm/` 这套 kernel 的推导与实现过程：每个设计决定是怎么得出的、
哪些数字是被交叉验证过的、哪些是尚未验证的假设。目的是让后续接手的人不必重新推一遍，
也能准确知道该在硬件上验哪几件事。

算法本身的定义见 [`bf16-dual-nvfp4-algorithm.html`](bf16-dual-nvfp4-algorithm.html)，
成品说明见 [`../README.md`](../README.md)。

---

## 0. 环境约束

实现在一台**没有 GPU、没有 nvcc、`3rdparty/cutlass` 子模块未初始化**的机器上完成
（torch 是 CPU 版）。这直接决定了工作方法：

- CUDA kernel 只能写，不能编译、不能跑。
- 因此凡是能在 CPU 上验证的部分，都必须真的验证——数值语义、资源预算算术、布局公式的自洽性。
- 凡是只能在硬件上确认的部分，必须**显式标注为假设**，而不是混在代码里当成已知事实。

第 8、9 节分别是"已验证的"和"未验证的"清单。

---

## 1. 从算法文档里提取的契约

先把文档压缩成 kernel 必须满足的几条硬约束：

| 契约 | 来源 | 对实现的影响 |
| --- | --- | --- |
| NVFP4 = E2M1 数据 + **每 16 元素** E4M3 块尺度 + 每 tensor FP32 全局尺度 | "真实 NVFP4 配置"一节 | MMA 必须是 block16 + E4M3，不是 MXFP4 的 block32 + UE8M0 |
| `G_A = 1`（activation `constant_amax` 固定为 2688 = 448×6） | expert input-scale1 recipe | 激活侧全局尺度在编译期消失，无参数、无指令 |
| `G_W` 保留到 epilogue，每 expert 一个 | 同上 | epilogue 需要一次 FP32 乘法；mainloop 不碰 |
| `s1 = s0/8`（derived_div8），零块把 `s0` 钳到 `8·2⁻⁹` | Algorithm 1 第 5 步 + 注释 | 钳位不是可选项，否则 `s1` 下溢为 0 → inf/NaN |
| 残差在**归一化域**算：`(x·rcp(s0) − q0)·8` | "Transform A 下钻"第 2 节 | 第二遍不付 dequant 代价；也决定了参考实现的算子顺序 |
| 两遍 MMA 共用同一份 `W`/`SFB`，落进**同一个**累加器 | Algorithm 2 第 10–11 行 | `A0×W` 用 `acc=(k≠0)`，`A1×W` 恒 `acc=true`，单次 epilogue |
| `A0/A1/SFA0/SFA1` 只存在于 producer stage，永不落 global | "关键不变量" | fused 路径不需要 workspace；SMEM 预算是主要矛盾 |

其中"`s0` 要用**舍入后**的值去建倒数"这一条容易漏。文档的"精确性边界"注释说得很清楚：
代码算的是 `x · rcp_approx(s0_rounded)`，不是 `x / (amax/6)`。参考实现和 kernel 都必须按这个顺序，
否则两者会在每个 `amax/6` 不能被 E4M3 精确表示的块上产生分歧。

---

## 2. DeepGEMM 骨架：能用什么，缺什么

选定 `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh` 作为骨架。
它的结构是 warp 特化 + 持久化 scheduler：

```
warp 0            TMA producer
warp 1            MMA issue（leader CTA）
warp 2            UTCCP transposer（把 TMA 来的 scale 转成 UTCCP 要的布局）
warp 3+           epilogue warp group
```

**可以直接复用的**（这些是 DeepGEMM 最有价值的部分，不重造）：

- `sched::Scheduler`——grouped/masked/contiguous 的持久化块调度与 L2 swizzle。
- 环形 stage 的 barrier 纪律：`full` / `empty` / `with_sf_full` 三组 + `advance_pipeline` 的
  `phase ^= (stage_idx == 0)` 翻转。
- **per-stage UMMA descriptor 的 lane 广播技巧**：lane `s` 的寄存器里存 stage `s` 的
  descriptor 低位，用一条 `exchange`（`__shfl_sync`）取出来，省掉每 stage 的地址重算。
- UTCCP scale 通路（`SM100_UTCCP_4x32dp128bit_1cta` + `make_sf_desc` + `replace_smem_desc_addr`）。
- epilogue 的 TMEM→寄存器→swizzled SMEM→TMA store 流水。

**缺的四块**：

1. **NVFP4 MMA 指令**。DeepGEMM 只有 `SM100_MMA_MXF4_SS`，即
   `kind::mxf4.block_scale.block32`（MXFP4：块 32、UE8M0 尺度）。NVFP4 是另一个 operand kind，
   要 `kind::mxf4nvf4.block_scale.block16`。→ `ptx/tcgen05_nvfp4.cuh`
2. **mainloop 内的 transform**。DeepGEMM 的 producer 只搬字节不算数。→ `transform/dual_nvfp4.cuh`
3. **每个 K 步两遍 MMA**共享一份 B 片段、共享累加器。→ 在 kernel 的 MMA warp 里
4. **epilogue 的 `G_W`**。DeepGEMM 的 `sm100_store_cd` 只有 `apply_index_n` 这种下标变换钩子，
   没有数值缩放钩子。→ `epilogue/sm100_store_cd_gscale.cuh`

另外发现一处 DeepGEMM 的工具函数**不能直接用**：`mma::sm100::advance_umma_desc_lo` 按
`sizeof(dtype_t)` 推进地址，但没有除以 pack factor。`make_umma_desc` 自己在算指针时除了
（`... * sizeof(dtype_t) / kPackFactor`），`advance_umma_desc_lo` 没有。对 E2M1 operand
会多推进一倍。所以 kernel 里自己写了 `advance_packed_fp4_desc_lo`，并在注释里说明原因。

---

## 3. 关键推导：scale factor 的布局

这是整个实现里最不确定、也最关键的一环，因为它同时决定 SMEM 尺寸、TMEM 列数、
UTCCP 调用次数和 MMA 的 TMEM 地址。推导路径如下。

### 3.1 从 DeepGEMM 的 transposer 反推 SMEM 布局

DeepGEMM 的 `utccp_required_smem_warp_transpose` 是：

```cpp
for (i = 0; i < 4; ++i) values[i] = ld_shared(smem_ptr + i * 32 + lane_idx);
st_shared(smem_ptr + lane_idx * 4, values[0..3]);
```

即 `new[j] = old[(j % 4) * 32 + j / 4]`，其中 `old` 按行号索引。所以 UTCCP 期望的 SMEM 布局里，
第 `j` 个 uint32 对应的行号是 `(j%4)*32 + j/4`。**反解得到**：给定行 `m`，它的 uint32 下标是

```
j = (m % 32) * 4 + m / 32
```

这就是 `transform/dual_nvfp4.cuh` 里 `sf_atom_word_idx()` 的来历。

由此得到一个直接的收益：**SFA 根本不需要 transposer**。DeepGEMM 必须转置是因为 scale 是 TMA
搬来的，布局由 global 决定；而我们的 SFA 是自己算出来的，可以直接按上式写成"已转置"的形态。
只有 SFB 还是 TMA 来的，仍需 warp 2。

### 3.2 用文档的数字反向验证 atom 尺寸

一个 UTCCP `4x32dp128bit` atom 搬 512 B。问题是这 512 B 在 NVFP4 下对应什么。

文档"让它藏住的设计杠杆"一节给了一组具体数字：

> TN=256 时 accumulator 256 列，SFA0/SFA1 各 8 列、SFB 16 列，共 288/512

拿它当方程解。假设 **1 个 atom = 128 行 × 4 个 E4M3 尺度 = 512 B → 4 个 TMEM 列**，
且 NVFP4 每 16 个 K 元素一个尺度，则一个 atom 覆盖 4×16 = 64 个 K 元素。于是对
`BLOCK_M=128, BLOCK_N=256, BLOCK_K=128`：

```
SFA 列数 = (128/128 行组) × (128/64 个 K atom) × 4 = 8    ✓ 文档说 8
SFB 列数 = (256/128 行组) × (128/64 个 K atom) × 4 = 16   ✓ 文档说 16
合计     = 256 + 8 + 8 + 16 = 288                          ✓ 文档说 288/512
```

三个数全部对上。这就把"128×4 atom"这个模型从猜测变成了有独立佐证的结论——文档另一处提到
hybrid 路径"直接写 FlashInfer/cuDNN 需要的 **128×4 scale layout**"，两边一致。

由此还顺带确定了 **`UMMA_K = 64`**：一条 `.block16` 指令消费一个完整 atom（4 个 16 元素子块），
所以每个 K 步推进 64 个元素、TMEM 列前进 4。

### 3.3 由此确定的尺寸

```
SMEM_SFA_SIZE_PER_STAGE = (BLOCK_K/64) × SF_BLOCK_M × 4 B   = 2 × 128 × 4 = 1024 B
SMEM_SFB_SIZE_PER_STAGE = (BLOCK_K/64) × SF_BLOCK_N × 4 B   = 2 × 256 × 4 = 2048 B
```

---

## 4. 关键推导：transform 的线程映射

文档的映射是：每条 lane 持有 8 个 BF16（半个 16 元素块），相邻 lane 用 butterfly shuffle
合并 amax。文档自己指出了这个映射的代价（"源码级成本：冗余在哪里"一节）：

> 两条 lane 虽然得到相同的 `s0 / rcp(s0) / s1`，但在 SIMT 上一条标量源码操作本来就是整个 warp
> 发射一条指令……因此这里是**数据结果重复，不等于 issued instruction 可减半**。

这个结论是对的——在给定映射下，predicate 掉一半 lane 并不能省指令。但它成立的前提是映射不变。
**换个映射就没有这份冗余可言。**

本实现让**每个线程持有一个完整的 64 元素 scale atom**：

| | 文档映射（半块/lane） | 本实现（整 atom/线程） |
| --- | --- | --- |
| amax | 7 fmax + 1 shfl + 1 fmax，每 8 元素 | 15 fmax，每 16 元素，无 shuffle |
| `s0`/`rcp`/`s1` | 每 8 元素算一次（成对重复） | 每 16 元素算一次 |
| SFA 写出 | 每块 2 次单字节写，两条 lane 同址同值 | 每 atom **1 次 4 字节写** |
| A0/A1 写出 | 8 B（半个 bank group） | 32 B = 2×`STS.128` |
| BF16 读入 | 16 B = 1×`LDS.128` | 128 B = 8×`LDS.128` |

归一到"每线程 64 个元素"比较：文档映射是 8 组 ×(7+1+1) = 56 fmax + 8 shfl，
且付 8 份标量尺度工作；本实现是 4 块 ×15 = 60 fmax + 0 shfl，付 4 份标量尺度工作。
逐元素工作量完全相同，省掉的是 shuffle 和一半的标量/写出开销。

选 64 而不是 32 元素还有两个具体理由：

- **64 = 一个 SF atom**，所以四个尺度正好凑成一个 uint32，SFA 写出是单条 4 字节存储；
  若取 32 元素，就是两次 2 字节存储。
- 64 个 BF16 = 128 B，正好是 128 B swizzle 的一整行 atom，寻址不跨 atom。

代价是每线程寄存器压力更高（64 个 float 的活跃区间）。在 `BLOCK_M=128, BLOCK_K=128`、
8 个 transform warp 下，任务数 = 128×2 = 256，线程数 = 256，**恰好每线程 1 个任务**，
不需要循环。

### 4.1 手工复现 swizzle

transform warp 用普通 `LDS` 读 BF16 A tile，没有 TMA 帮它解 swizzle，所以必须自己复现。
`swizzled_byte_offset()` 实现的映射是：

```
atom_idx   = k_byte / kSwizzleMode              // TMA 把一行切成若干 swizzle atom
bank_group = (k_byte % kSwizzleMode) / 16
offset = atom_idx * (BLOCK_MN * kSwizzleMode)   // atom 子 tile
       + m * kSwizzleMode                        // atom 内的行
       + (bank_group ^ (m % (kSwizzleMode/16))) * 16 + k_byte % 16
```

与 DeepGEMM `tma::copy` 的循环 `smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM` 一致
（atom 子 tile 的 stride = `BLOCK_MN × atom 行字节`）。

- BF16 A：`kSwizzleAMode = 128`，一行 256 B → 2 个 atom，`^ (m % 8)`
- packed FP4 A0/A1/W：`kSwizzleABMode = 64`，一行 64 B → 1 个 atom，`^ (m % 4)`

后者满足 `make_umma_desc` 的断言 `kSwizzleMode × kPackFactor == BLOCK_K × sizeof(dtype)`
即 `64×2 == 128×1` ✓。

**这是最容易出错且失败最隐蔽的一处**：swizzle 复现错了不会报错，只会算出错误结果。
第 9 节给了单独验证它的方法。

---

## 5. 流水线与正确性论证

在 DeepGEMM 三组 barrier 的基础上，把 `with_sf_full` 改成语义更宽的 `transform_full`：

```
full_barriers[s]            TMA 完成（A_bf16 + W + SFB 落 SMEM），init(1)
transform_full_barriers[s]  A0/A1/SFA0/SFA1 写完 且 SFB 转置完
                            init(kNumTransformThreads + 32)
empty_barriers[s]           MMA 消费完该 stage（umma_arrive），init(1)
```

数据流：`TMA → {transform warps, SFB transposer} → MMA → epilogue`。

**竞争检查**（三条都必须成立）：

1. *transform 会不会读到还没搬完的 A？* 不会——transform 等 `full_barriers[s]`。
2. *TMA 会不会覆盖 transform 还在读的 `smem_a[s]`？* 不会——TMA 等 `empty_barriers[s]`，
   而该 barrier 由 MMA 释放，MMA 又等 `transform_full[s]`，即 transform 已完成。
3. *MMA 会不会读到写了一半的 `A0/A1`？* 不会——MMA 等 `transform_full[s]`，
   该 barrier 要求全部 256 个 transform 线程都 arrive。

这里 `smem_a[s]` 和 `smem_a0/a1[s]` 共用同一个 stage 索引和同一个 empty barrier。
这偏保守——BF16 A 本可以在 transform 完成后就立即释放，不必等 MMA。用统一 stage 换取了
逻辑简单；第 10 节把解耦列为后续优化。

transform warp 写完必须 `fence_view_async_shared()` 再 arrive，因为 UTCCP/UMMA 是通过
async proxy 读 SMEM 的，普通 `STS` 对它们不自动可见。

---

## 6. 两遍 MMA 的发射

每个 K tile，MMA warp 做：

```
UTCCP:  SFA0 的 kNumKAtoms×kNumSFASubAtoms 个 atom → TMEM
        SFA1 同上
        SFB  的 kNumKAtoms×kNumSFBSubAtoms 个 atom → TMEM
for a in 0..kNumKAtoms-1:              # a 即 UMMA-K 步，每步 64 个 K 元素
    mma(A0+a·32B, W+a·32B, SFA0+a·4列, SFB+a·8列, acc = (a>0 或 k_block>0))
    mma(A1+a·32B, W+a·32B, SFA1+a·4列, SFB+a·8列, acc = true)
```

注意 `acc=false` 只出现在**整个 K 循环的第一条指令**上，之后全部累加——这正是文档
"累加寄存器共享"不变量的要求。两遍共用 `desc_b`（同一份 W 片段），逻辑 B 流量保持单份。

`kNumEpilogueStages` 不能像 DeepGEMM 那样固定为 2：dual-A 多占了 SFA0/SFA1 两组列，
`BLOCK_N=256` 时两级累加器要 512+32 = 544 列，超了。所以改成推导式：

```cpp
kNumEpilogueStages = (UMMA_N * 2 + kNumSFTmemCols <= 512) ? 2 : 1;
```

`128×256×128` 得 1 级，`128×128×128` 得 2 级。

---

## 7. 资源预算核算

用独立脚本按 kernel 里的 constexpr 逐项复算，结果：

| tile | TMEM | SMEM/stage | stages | 总 SMEM | epilogue stages | transform 任务/线程 |
| --- | --- | --- | --- | --- | --- | --- |
| **128×256×128** | 256+8+8+16 = **288/512** | 68.0 KB | **2** | 168.1 / 227 KB | 1 | 1.0 |
| 128×128×128 | 256+8+8+8 = 280/512 | 59.0 KB | 3 | 209.1 / 227 KB | 2 | 1.0 |
| 128×256×256 | 256+16+16+32 = 320/512 | 136.0 KB | 1 | 168.0 / 227 KB | 1 | 2.0 |

`288/512` 与文档一致（第 3.2 节）。

SMEM/stage 的 68 KB 拆开是：A(BF16) 32 KB + A0 8 KB + A1 8 KB + W 16 KB + 尺度 4 KB。
**BF16 的 A 独占了半个 stage**，这是 dual-A 相对普通 NVFP4 GEMM 最大的 SMEM 代价，
也是流水只能开到 2 级的直接原因。`128×256×256` 只能开 1 级 stage（等于没有流水），
所以不作为候选。

---

## 8. 数值参考与已验证的事项

`tests/reference/dual_nvfp4.py` 是 Algorithm 1/2 的**可执行规格**，逐算子对齐 kernel 的顺序：
E4M3 往返、`s1` 二次舍入、归一化域残差、E2M1 就近偶数舍入 + 饱和到 ±6。
唯一有意偏离的是 `rcp_approx`（CPU 没有等价物，默认用精确除法）。

`tests/test_dual_nvfp4_reference.py` 全部通过，验证了：

- E2M1 格点是不动点；`±6.5/1e4/inf` 饱和到 ±6；七个中点全部按偶数舍入
  （`0.25→0`、`0.75→1`、`2.5→2`、`3.5→4`、`5.0→4`）。
- **零块**：`s0` 被钳住、`s1 > 0` 且有限、重构恰好为 0。这是 `derived_div8` 钳位的直接回归测试。
- 已在格点上的块单遍即精确重构。
- 残差遍把相对 L2 从 0.09519 降到 0.01205（**7.9×**）。
- `residual_amax` 不劣于 `derived_div8`。
- 端到端 grouped GEMM 对 ground truth 的 cosine = **0.999926**。

以及资源预算算术（第 7 节）和布局公式的自洽性（第 3、4 节）。

### 8.1 发现的文档不一致

文档的 "Ground truth" 一节明确定义：

> `C_ref = A_bf16 · (dec(W_fp4)·S_W·G_W)ᵀ`（FP32 累加），ground truth 必须用 BF16 原值，
> **不能**引入任何 FP8 中间量化，否则会低估真实误差。

但它报告的 fused cosine 0.995396 / hybrid 0.995407 与这个定义对不上。实测三条基准：

| 比较 | cosine |
| --- | --- |
| 仅权重量化：`A_bf16×dequant(W)` vs `A_bf16×W_bf16` | 0.995438 |
| **dual-A vs 文档定义的 ground truth** | **0.999926** |
| dual-A vs 全 BF16 | 0.995366 |
| 单遍 NVFP4 激活 vs 文档定义的 ground truth | 0.995417 |

文档报告的数值对应的是**第三行**（对全 BF16 比），此时误差被权重量化主导，
所以 dual-A(0.995366)、单遍(0.995417)、纯权重量化(0.995438) 三者几乎无法区分。

**这有实际后果**：如果拿 0.995 当 kernel 的验收阈值，一个第二遍完全失效（比如 `s1` 算错、
`A1` 的 MMA 没累加进去）的 kernel 也能通过。正确的阈值是 **0.999+**，
对着文档自己定义的 ground truth 比。

---

## 9. 未验证的假设与验证方法

kernel **未经编译、未经运行**。以下五点是代码里编码了、但离线无法确认的假设，
按"错了以后有多难发现"排序：

| # | 假设 | 位置 | 怎么验 |
| --- | --- | --- | --- |
| 1 | swizzle 手工复现正确 | `swizzled_byte_offset()` | **最该先做、也最容易单独做**：TMA 一个已知 pattern 进 A stage，只跑 transform，把 A0/SFA0 拷回 host，与 `tests/reference/dual_nvfp4.py` 逐元素比。错了不会报错，只会算错。 |
| 2 | NVFP4 的 TMEM SF 寻址 | MMA warp：每 UMMA-K 步 TMEM 列 +4、`sf_id = 0` | 对照 PTX ISA 中 `tcgen05.mma ... .block16` 对 scale operand 的描述。模型是"一条 `.block16` 恰好消费一个 128×4 atom"。 |
| 3 | packed FP4 的 SMEM 形态 | transform 写 packed；host 用 `fp4_unpacked_smem=false` | DeepGEMM 自己的 FP4 路径用的是 **unpacked** SMEM 变体（`float_e2m1_unpacksmem_t`）。需确认 UMMA 能按 64 B swizzle 读真正打包的 SMEM。若不能，transform 的写出格式要跟着改。 |
| 4 | `cvt.rn.satfinite.e2m1x2.f32` 的操作数顺序 | `ptx/tcgen05_nvfp4.cuh` | 当前假设第一个参数进高 nibble。一条指令即可验：转一对已知值，读回字节。错了表现为块内相邻元素两两互换。 |
| 5 | `cutlass::float_ue4m3_t` 是 NVFP4 的正确 SF 类型 | `make_instr_desc_block_scaled` | 初始化 `3rdparty/cutlass` 后 grep 即可确认；编译期就会暴露。 |

第 5 条编译就能发现，第 4 条一条指令就能验，第 1 条有明确的隔离测试路径。
第 2、3 条需要读 PTX ISA 文档或在硬件上试。

`3rdparty/cutlass` 子模块当前未初始化（拉取时代理 `42.192.60.90:31260` 中途 TLS 断连、
重试被拒），编译前需要先补上。

---

## 10. 后续方向

按预期收益排序：

1. **跨 N tile 复用 A（结构性）**。文档的测量对此毫不含糊：fused 0.1142 ms vs
   hybrid 0.0683 ms（约 **1.67×**）。fused 路径对每个 N tile 重算同一份 transform，
   这是主要放大项。hybrid 用一块 global workspace 换成整个 A 只 transform 一次。
   本实现走的是 fused 路径，所以这个放大项还在。
2. **解耦 A 与 A0/A1 的流水**。BF16 A 占 68 KB stage 中的 32 KB，且它的消费者是
   transform warp，而 A0/A1 的消费者是 MMA。分开 stage 后，受 HBM 延迟约束的 A 通路
   可以开到 3–4 级，而不必为 FP4 缓冲付同样的深度。这是当前 2 级流水的直接解法。
3. **transform warp 数与 epilogue 复用的取舍**。文档实测复用 epilogue warp 把 0.172 ms
   降到 0.1395 ms；但复用会让 block *i* 的 epilogue 与 block *i+1* 的 mainloop 串行化，
   而持久化 scheduler 本来能重叠这两者。本实现选了独立 warp group（保留重叠，多 128 线程），
   哪边划算取决于形状，需要实测。这是一个模板参数的距离。
4. **2-CTA multicast**。当前 `kNumMulticast == 1`。2-SM UMMA 下 producer warp 写的 operand
   要被 peer CTA 的 MMA 消费，这是真正的扩展而非参数开关，所以没有做半截。
