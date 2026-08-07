# Dual-NVFP4 Fused Grouped GEMM on B300 — Kernel Analysis

主要资料:本仓库 `include/nvfp4_gemm/` 与 `src/`;实测账本 [b300-deep-opt-log.html](b300-deep-opt-log.html);时序图 [transform-warp-timeline-r36.drawio](transform-warp-timeline-r36.drawio) / [r37](transform-warp-timeline-r37.drawio) / [r38](transform-warp-timeline-r38.drawio) / [split](transform-warp-timeline-split.drawio)。模板参照 *Kimi K3 Inference Kernel Analysis: High-Performance Kernels & MXFP4 QAT*。

任务形态:Qwen3.5-35B-A3B 的 MoE expert GEMM(EP=8,本卡 32 local experts,top-8 routing),m-grouped contiguous 布局:

$$D_{[e]} = A_{[e]} W_e^\top,\qquad A \in \mathbb{R}^{M\times K}\ (\text{BF16 激活}),\quad W_e \in \mathbb{R}^{N\times K}\ (\text{NVFP4 预量化权重})$$

| 家族 | N | K | decode tokens | prefill tokens |
|---|---|---|---|---|
| gate_up | 1024 | 2048 | 16 / 32 / 128 | 2k / 8k / 16k |
| down | 2048 | 512 | 16 / 32 / 128 | 2k / 8k / 16k |

设计契约:**单 kernel、BF16 激活进、量化在核内 fused**——A 的双 NVFP4 分解逐 tile 在片上完成,分解产物从不落全局显存。

---

## 1. 数学推导:双 NVFP4 分解(Algorithm 1 的来历)

### 1.1 NVFP4 与单遍量化误差

NVFP4 = E2M1 元素 + 每 16 元素一个 E4M3 block scale。E2M1 的可表示网格:

$$\mathcal{G} = \{0,\ \pm 0.5,\ \pm 1,\ \pm 1.5,\ \pm 2,\ \pm 3,\ \pm 4,\ \pm 6\}$$

网格在 $[0,4]$ 内最大间隙为 1,在 $(4,6]$ 间隙为 2。对一个 16 元素块,记 $M = \max_i |x_i|$(块 amax),单遍 NVFP4 取

$$s_0 = Q_{\mathrm{e4m3}}\!\big(M/6\big),\qquad u_i = x_i / s_0 \in [-6,6],\qquad q^{(0)}_i = Q_{\mathrm{e2m1}}(u_i)$$

RN 舍入下 $\max_i |u_i - \mathrm{dec}(q^{(0)}_i)| \le 1$(最坏点在 4–6 间隙中点 $u=5$),故单遍逐元素最坏误差

$$\big|x_i - s_0\,\mathrm{dec}(q^{(0)}_i)\big| \;\le\; s_0 \cdot 1 \;=\; \underbrace{M/6}_{\text{单遍最坏}}$$

最小修正粒度(非零最小网格 0.5)为 $s_0\cdot 0.5 = M/12$。

### 1.2 双遍分解:残差再量化

Algorithm 1 把同一块再量化一次残差:

$$r_i = \big(u_i - \mathrm{dec}(q^{(0)}_i)\big)\times R,\qquad q^{(1)}_i = Q_{\mathrm{e2m1}}(r_i),\qquad s_1 = s_0 / R,\quad R = 8$$

重构值 $\hat{x}_i = s_0\,\mathrm{dec}(q^{(0)}_i) + s_1\,\mathrm{dec}(q^{(1)}_i)$。误差分解:

$$x_i - \hat{x}_i \;=\; \underbrace{s_0\big(u_i - \mathrm{dec}(q^{(0)}_i)\big)}_{=\,s_0 r_i / R} - \;\frac{s_0}{R}\,\mathrm{dec}(q^{(1)}_i) \;=\; \frac{s_0}{R}\big(r_i - \mathrm{dec}(q^{(1)}_i)\big)$$

$R=8$ 时 $r_i \in [-8,8]$,超出 E2M1 满量程 6 的部分被 `satfinite` 裁剪:

- 未裁剪区($|r_i|\le 6$):$|r_i - \mathrm{dec}(q^{(1)}_i)| \le 1$ → 误差 $\le s_0/8 = M/48$
- 裁剪区($|r_i| \in (6,8]$):$|r_i - 6| \le 2$ → 误差 $\le 2\,s_0/8 = \boxed{M/24}$(最坏)

最小修正粒度降为 $s_1 \cdot 0.5 = M/96$ —— 比单遍细 **8×**。这就是双遍的收益本质:同一 TMEM 累加器上,两遍 UMMA 各用自己的 $(q, s)$ 精确求和(FP32 累加):

$$D \;=\; \sum_k \big(s_0\,\mathrm{dec}(q^{(0)}) + s_1\,\mathrm{dec}(q^{(1)})\big)\cdot s_w\,\mathrm{dec}(w)$$

实测(2k gate_up,vs FP32 参考):cosine **0.9954**(dual)/ 0.9910(single)/ 0.9546(mxfp8×mxfp4)——dual 误差为 single 的 1/2、mxfp8 的 1/10。

**$(T,R)$ 参数面**($s_0 = Q_{\mathrm{e4m3}}(M/T)$;离散格点扫描验证,R40 实现并实测):

| 配置 | $\max\|e_0\|$ | 第二遍裁剪 | 最坏误差 | 最小修正量 |
|---|---|---|---|---|
| $T{=}6, R{=}8$(默认) | 1 | 有($r\le 8$) | $M/24$ | $M/96$ |
| $T{=}6, R{=}4$ | 1 | 无 | $M/48$ | $M/48$ |
| $T{=}4, R{=}8$ | 1/2 | 无($r\le 4$) | $M/64$ | $M/64$ |

R40 判决:$T{=}4$ 的 worst-case 改善 2.67× **不传导到 GEMM 级指标**(cos ±7e-5,两种数据分布)——点积误差由方差主导,$T{=}6$ 更细的最小修正量($M/96$ vs $M/64$)在典型数据上抵消尾部收益。默认保持 $T{=}6,R{=}8$;`scale_policy="div4_radix8"` 为零成本候选开关。

### 1.3 三个实现级恒等式(链上省指令的数学依据)

**(i) $s_0$ 的整数解码**。floor 使 $s_0 \ge 2^{-6}$(E4M3 最小 normal),normal E4M3(S=0)值为 $2^{E-7}(1 + m/8)$。其 FP32 位型的指数域 $= E - 7 + 127$、尾数域 $= m \ll 20$,合并即

$$\mathrm{bits}_{f32}(s_0) = (\mathrm{code} \ll 20) + \underbrace{\texttt{0x3C000000}}_{(127-7)\ll 23}$$

一次 IMAD 替代硬件 `cvt` 往返([dual_nvfp4.cuh:408](../include/nvfp4_gemm/transform/dual_nvfp4.cuh#L408))。floor $2^{-6}$ 同时保证 $s_1 = s_0/8 \ge 2^{-9}$(E4M3 最小正 subnormal):全零块给出 $q^{(0)}{=}q^{(1)}{=}0$ 与有限 scale,不产生 inf/NaN。

**(ii) 残差减法的 Sterbenz 精确性**。$\mathrm{dec}(q^{(0)}_i)$ 是 $u_i$ 的最近网格点,对非零 $u_i$ 恒有 $\tfrac{1}{2}\mathrm{dec} \le u_i \le 2\,\mathrm{dec}$,Sterbenz 引理给出 $u_i - \mathrm{dec}(q^{(0)}_i)$ 在 BF16 中**精确**(无舍入);$u_i = 0$ 平凡精确。因此残差链不引入额外误差。

**(iii) ×8 = 指数域 +3**。BF16 位型上乘 $2^3$ 即指数域加 3,对打包的 `bf16x2` 一条整数加实现:

$$r_{2i,2i+1} = \mathrm{bits}(u - \mathrm{dec}) + \underbrace{\texttt{0x01800180}}_{(3\ll 7)\,\|\,(3\ll 7)\ll 16}$$

$u-\mathrm{dec}= \pm 0$ 时加出的 $\pm 2^{-124}$ 级 denormal 被后续 `cvt.e2m1` 舍入回 0——含零逐位等价([dual_nvfp4.cuh:418-419](../include/nvfp4_gemm/transform/dual_nvfp4.cuh#L418))。

### 1.4 记号 ↔ 源码对照

| 本文记号 | 源码([dual_nvfp4.cuh](../include/nvfp4_gemm/transform/dual_nvfp4.cuh)) | 说明 |
|---|---|---|
| $M$ | `amax`(`__hmax2`/`__habs2` 树) | 块 amax,VHMNMX.BF16 原生 |
| $1/T$ | `kInvS0Div<policy>`(默认 $1/6$) | $s_0$ 除数;`Div4Radix8` 时 $1/4$ |
| $s_0$ | `s0_code` / `s0f` | E4M3 码 / 整数解码 FP32 |
| $u_i$ | `u2[]` / `ua[]`(bf16x2 打包) | `__hmul2(x, rcp(s0))` |
| $Q_{\mathrm{e2m1}}$ | `ptx::cvt_e2m1x8_bf16x2` | `cvt.rn.satfinite.e2m1x2.bf16x2` ×4 对/asm |
| $\mathrm{dec}$ | `ptx::cvt_bf16x2_e2m1x2` | `cvt.rn.bf16x2.e2m1x2`(sm_100f,CUDA 头文件未包) |
| $\times R$ | `kBump2 = 0x01800180` | 指数域 +3 |
| $s_1 = s_0/R$ | `s1_code`,`kResidualRadix` | E4M3 |
| $q^{(0)}, q^{(1)}$ | `q0w[]`, `q1w[]` → SMEM `A0/A1` | 打包 E2M1,UMMA 直接消费 |

---

## 2. Algorithm:与代码 1:1

### Algorithm 1: 双块交错分解 — `decompose_block_packed_x2`([dual_nvfp4.cuh:359-463](../include/nvfp4_gemm/transform/dual_nvfp4.cuh#L359-L463))

每线程持有 2 个独立 16 元素块(§3 的 slot 映射),两条链显式交错——两个 E4M3 往返 + MUFU rcp(最长延迟环节)先行流水,B 块的标量延迟藏进 A 块的元素运算。与逐块形态同运算同舍入序,输出逐位一致。

$$
\begin{array}{r l l}
\hline
& \rlap{\textbf{Algorithm 1: } \texttt{decompose\_block\_packed\_x2}\ \text{ —— 每线程 2 块(32 元素),两链显式交错}} &\\
\hline
& \rlap{\textbf{输入:}\ x_a[8],\ x_b[8]\ \text{(各 8×u32 = 16 BF16,LDS.128 已载入)}} &\\
& \rlap{\textbf{输出:}\ q_a^{(0)},q_b^{(0)},q_a^{(1)},q_b^{(1)}\ \text{(各 2 字打包 E2M1,8 码/字)};\ \ s_{0a},s_{0b},s_{1a},s_{1b}\ \text{(E4M3 标量码)}} &\\
\hline
& \rlap{\text{—— Phase 1:两棵 amax 树,交错(L379–403)}} &\\
1 & t_a[i] \leftarrow \mathrm{hmax2}\big(\mathrm{habs2}(x_a[2i]),\ \mathrm{habs2}(x_a[2i{+}1])\big),\ \ i<4 & \triangleright\ \text{深度 3 树,VHMNMX 原生}\\
2 & t_b[i] \leftarrow \text{同上} & \triangleright\ \text{原线性链深 7,R31 改树}\\
3 & \text{树归约 stride } 2 \to 1;\ \texttt{byte\_perm}\ \text{半字交换再 hmax2} \Rightarrow M_a,\ M_b & \triangleright\ \text{块 amax,精确}\\
& \rlap{\text{—— Phase 2:两条标量链(两个 MUFU rcp 互相流水)(L404–416)}} &\\
4 & s_0 \leftarrow \mathrm{cvt\_e4m3}\big(\max(M \cdot \texttt{kInvS0Div},\ 2^{-6})\big) & \triangleright\ \text{§1.1 } s_0\text{;floor §1.3(i)}\\
5 & s_0^{f32} \leftarrow \mathrm{int\_as\_float}\big((s_0 \ll 20) + \texttt{0x3C000000}\big) & \triangleright\ \text{§1.3(i) 整数解码}\\
6 & \mathrm{inv} \leftarrow \mathrm{rcp}(s_0^{f32});\quad s_1 \leftarrow \mathrm{cvt\_e4m3}(s_0^{f32}/8) & \triangleright\ \text{rcp 消融无罪(≤0.2µs)}\\
& \rlap{\text{—— Phase 3:元素运算,A/B 字交错(L417–449);循环 } w<2,\ j<4} &\\
7 & u[j] \leftarrow \mathrm{hmul2}(x[4w{+}j],\ \mathrm{inv2}) & \triangleright\ \text{无原生 bf16x2 乘:2×FMUL+F2FP(R25)}\\
8 & q^{(0)}[w] \leftarrow \mathrm{cvt\_e2m1x8}(u[0..3]) & \triangleright\ \text{4 对/asm 向量打包量化}\\
9 & d[j] \leftarrow \mathrm{cvt\_bf16x2}(q^{(0)}[w] \gg 8j) & \triangleright\ \mathrm{dec}(q^{(0)})\text{,硬件解码}\\
10 & r[j] \leftarrow \mathrm{bits}\big(\mathrm{hsub2}(u[j],\ d[j])\big) + \texttt{0x01800180} & \triangleright\ \text{Sterbenz 精确 + ×8,§1.3(ii,iii)}\\
11 & q^{(1)}[w] \leftarrow \mathrm{cvt\_e2m1x8}(r[0..3]) & \triangleright\ \text{残差遍}\\
\hline
\end{array}
$$

**函数体对照表**:

| 行 | 语句 | 对应推导 | 说明 |
|---|---|---|---|
| 1-3 | `__hmax2(__habs2(...))` 树 | $M=\max\|x_i\|$ | 精确(max 对结合律不敏感);深 7→3 实测 −0.3µs |
| 4 | `cvt_e4m3(fmaxf(amax*kInvS0Div, kFloor))` | $s_0 = Q_{e4m3}(M/6)$,floor | `cvt.rn.satfinite.e4m3x2` |
| 5 | `(s0c<<20)+0x3C000000` | §1.3(i) | 一条 IMAD |
| 6 | `math::fast_rcp` | $1/s_0$ | MUFU;exact-bit 变体实测中性 |
| 7-8 | `__hmul2` ×4 + `cvt_e2m1x8_bf16x2` | $q^{(0)}=Q_{e2m1}(u)$ | 打包量化,1 asm/8 码 |
| 9-10 | `cvt_bf16x2_e2m1x2` + `__hsub2` + `+kBump2` | $r=(u-\mathrm{dec})\times 8$ | 精确减 + 指数 bump |
| 11 | `cvt_e2m1x8_bf16x2` | $q^{(1)}=Q_{e2m1}(r)$ | 双遍完成 |

### Algorithm 2: tile 变换与 slot 映射 — `transform_a_tile`([dual_nvfp4.cuh:488-660](../include/nvfp4_gemm/transform/dual_nvfp4.cuh#L488-L660))

线程→数据映射的原则:**每线程恰好持有一个 128×4 scale atom 的整数分之一 K 段**(64 连续元素的 1、1/2 或 1/4),slot 不跨 atom:

$$\texttt{kBlocksPerSlot} = \min\!\Big(\frac{\text{BLOCK\_M}\cdot\text{BLOCK\_K}/16}{\text{线程数}},\ 4\Big) \in \{4,2,1\}$$

默认 prefill(bm128/bk128/tw16=512 线程)→ 2 块/线程(半 atom),走 Algorithm 1 的 pair 路径;swap decode(bk256/tw8)→ 4 块/线程,逐块路径(pair 变体的寄存器占用在该档实测 +2µs)。收益:块 scale 落在一次对齐 SF 存储(u32/u16/u8),amax 无需任何 `shfl`,无标量重复计算。

- LDS:`ld.shared.v4`(128-bit)按 CuTe `Swizzle<3,4,3>` 字节偏移读 A([:533-545](../include/nvfp4_gemm/transform/dual_nvfp4.cuh#L533-L545))
- 存储:A0/A1 `st.shared.v4`(整/半 atom)或 `st.shared.v2.b32`(1/4 atom);SF 按 UTCCP 期望布局**预转置**写入,$\mathrm{word}(m) = (m \bmod 32)\cdot 4 + \lfloor m/32\rfloor$([:102-105](../include/nvfp4_gemm/transform/dual_nvfp4.cuh#L102-L105)),省掉 DeepGEMM 的 warp shuffle 转置

### Algorithm 3: 主 kernel,warp 特化角色 —— [sm100_bf16_dual_nvfp4_gemm.cuh](../include/nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm.cuh)

$$
\begin{array}{r l l}
\hline
& \rlap{\textbf{Algorithm 3: } \texttt{sm100\_bf16\_dual\_nvfp4\_gemm}\ \text{ —— persistent,warp 特化,主循环零 } \texttt{\_\_syncthreads}} &\\
\hline
& \rlap{\text{序幕(全 CTA):warp1 init 22 个 mbarrier(L273–295);warp2 分配 TMEM 512 列;}\texttt{\_\_syncthreads}\text{;PDL 同步}} &\\
& \rlap{\text{各角色主体均为 for 每 k-block } t \text{ 的循环,仅经 mbarrier 通信;下列为循环体}} &\\
\hline
1 & \textbf{if}\ \text{warp} = 0\ \text{(TMA producer,L341–440)}: & \triangleright\ \text{wait 80\% / work 7\%}\\
2 & \quad \mathrm{try\_wait}(\mathrm{empty}[s], \varphi{\oplus}1);\ \text{影子门 } \mathrm{tfm\_full}[t{-}3] & \triangleright\ \text{B 槽释放 + A 槽读完}\\
3 & \quad \text{cp.async.bulk.tensor} \times 3:\ A\,(32768\,\mathrm{B}) \to \mathrm{full\_a}[s];\ W{+}\mathrm{SFB}\,(18432\,\mathrm{B}) \to \mathrm{full}[s] & \triangleright\ \texttt{arrive\_and\_expect\_tx}\\
4 & \quad \text{尾:}\ \text{cudaTriggerProgrammaticLaunchCompletion(PDL)} &\\
5 & \textbf{elif}\ \text{warp} = 1\ \text{(MMA issue,L445–636)}: & \triangleright\ \text{wait 64\% / work 31\%}\\
6 & \quad \mathrm{try\_wait}(\mathrm{tfm\_full}[s], \varphi);\ \mathrm{try\_wait}(\mathrm{full}[s], \varphi) & \triangleright\ 544\ \text{arrive;B 快门}\\
7 & \quad \text{exchange } a_0/a_1\ \text{描述符(} \mathrm{slot}_{pr}\ \text{游标);elect\_one: UTCCP 拷 SFA0/SFA1/SFB} \to \text{TMEM} &\\
8 & \quad \textbf{for}\ \text{每 64-K atom:}\ \text{tcgen05.mma}(A_0{\times}W,\ \text{首块 } \mathrm{scale\_c}{=}0);\ \ \text{tcgen05.mma}(A_1{\times}W,\ \text{恒累加}) & \triangleright\ \text{双遍同一累加器,§1.2}\\
9 & \quad \mathrm{umma\_arrive}(\mathrm{empty}[s]);\ \text{tile 尾块再} \to \mathrm{umma\_arrive}(\mathrm{tmem\_full}) &\\
10 & \textbf{elif}\ \text{warp} = 2\ \text{(L642–675)}:\ \text{SFB warp-shuffle 转置} \to \mathrm{arrive}(\mathrm{tfm\_full}[s]) & \triangleright\ \text{32 arrive 份额}\\
11 & \textbf{elif}\ \text{warp} \in [3,19)\ \text{(transform ×16,L679–786)}: & \triangleright\ \text{wait 22\% / work 76\%}\\
12 & \quad \mathrm{try\_wait}(\mathrm{full\_a}[s], \varphi);\ \mathrm{try\_wait}(\mathrm{empty}[s_{pr}], \varphi_{pr}) & \triangleright\ \text{A 到货 + 产物槽空}\\
13 & \quad \texttt{transform\_a\_tile}\ \text{(Algorithm 2 → Algorithm 1)} & \triangleright\ 575\,\mathrm{ns}\text{,节拍源}\\
14 & \quad \text{fence.proxy.async};\ \mathrm{arrive}(\mathrm{tfm\_full}[s]) & \triangleright\ 512\ \text{线程计数}\\
15 & \textbf{else}\ \text{(epilogue ×4 warp,L790+)}: & \triangleright\ \text{wait 87\% / work 13\%(per tile)}\\
16 & \quad \text{每 16 拍被尾块 umma\_arrive 唤醒:tcgen05.ld 批量读 TMEM} \to {\times}G_W \to \text{BF16} & \\
17 & \quad \to \text{TMA store(cd64 双缓冲)} \to \mathrm{arrive}(\mathrm{tmem\_empty},\ 128\ \text{线程}) &\\
\hline
\end{array}
$$

wait/work 占比为 probe3 warp 时钟实测(8k gate_up,own-iter 归一,GHz 1.965):tma 0.603/0.051、tfm 0.168/0.570、mma 0.486/0.236 µs/kb,epi 12.15/1.75 µs/tile。

---

## 3. 真实 Pipeline

### 3.1 执行配置

| 项 | 值 | 说明 |
|---|---|---|
| grid | 148 CTA,persistent | 静态跨步 scheduler;**sms=107 == 148 实测等速**(41 SM 全程零贡献,可让给并发流) |
| block | 736 thread = 23 warp | warp0 TMA、warp1 MMA、warp2 SFB 转置、warp3-18 transform×16、warp19-22 epilogue |
| bounds | `__launch_bounds__(736, 1)` | 1 CTA/SM(SMEM 占满) |
| SMEM | 225 464 B / CTA | A 32K×**3** + 产物 18K×**2** + W/SFB 18K×**4** + CD 16K + barriers 184B(异深深度见 §3.3) |
| barrier | 22 mbarrier + 2 `__syncthreads` | 5 类×4 逻辑 stage + 2 类×1 epi stage;`__syncthreads` 仅 init/收尾 |
| TMEM | 512 列 | UMMA_N=256 accumulator ×1 + SFA0/SFA1/SFB 列组 |
| 启动 | PDL on,cluster 1 | JIT 按 (shape, config, policy, probe) 特化,cache 按头文件内容哈希 |

### 3.2 mbarrier 协议(所有交接的底层)

| barrier | ×数 | 完成条件(init) | 生产者 → 等待者 | 语义 |
|---|---|---|---|---|
| `full` | 4 | tx-count:W 16 384 + SFB 2 048 B | TMA engine → warp1/warp2 | B/SFB 到货 |
| `full_a` | 4 | tx-count:A 32 768 B | TMA engine → transform | A 到货(A-first,R36) |
| `empty` | 4 | 1 × `umma_arrive` | Tensor Core → warp0 | B 槽读完可覆写 |
| `transform_full` | 4 | **544** arrive(512 tfm + 32 warp2) | transform+warp2 → warp1(warp0 走影子门 t−3) | 产物 + SFB 转置就绪 |
| `a0_full` | 4 | 默认不使用(split 模式 544) | — | 半程流水,存档负收益 |
| `tmem_full` | 1 | 1 × `umma_arrive`(tile 尾块) | warp1 → epilogue | 累加器完成 |
| `tmem_empty` | 1 | 128 arrive | epilogue → warp1 | 累加器腾空 |

数据类屏障按 **tx-count** 计字节;线程类按 **arrive 计数**;每次完成 phase 位翻转,等待方 `mbarrier.try_wait(旧 parity)` 硬件挂起(不占发射口),翻转即醒(~60ns)。复用轮换即代码里的 `phase ^= (stage == 0)`。

### 3.3 稳态时序与环模型

**环模型**(三路验证):稳态节拍 = stage 生命周期环(transform + 发射 + exec + TMA tx + 唤醒 ≈ 2.5µs @bk128)÷ 流水深度。V3 链缩短的收益按 Δ/3 兑现;stage=2 时坍缩为 环/1;TMA 93% idle 因为 tx 在环内。

**异深深度**(R35,SMEM 预算的关键腾挪):各缓冲按真实生命周期取深——A×**3**(影子门 `transform_full[t−3]` 释放)、产物 A0/A1/SFA×**2**(UMMA 读完即 `empty[t−2]` 窗)、W/SFB×**4** 不可浅(与 `empty` 同一释放窗,浅 SFB 会重建整环)。槽游标 = `t mod 深度`(**绝不是** `stage mod 深度`)。省出的字节买回 cd64 双缓冲 epilogue。

**A-first 到达**(R36):transform 只需要 A,给 A 单独的 `full_a`(TMA 先发 A,DMA 引擎串行——B-first 反序实测 +0.6µs 证明),W/SFB 的 tx 尾巴从变换关键链上摘除。

**单拍分解**(8k gate_up,period 754 ns/kb,图见 [r36 时序图](transform-warp-timeline-r36.drawio)):

$$\underbrace{60}_{\text{唤醒}} + \underbrace{575}_{\text{Algorithm 1}} + \underbrace{60}_{\text{唤醒}} + \underbrace{290}_{\text{UTCCP+发射}} \;\text{对抗}\; \underbrace{496}_{\text{exec 背靠背}} \;\Rightarrow\; \text{idle} = 754 - 496 = 258\ \mathrm{ns/kb}$$

### 3.4 流水线演进(每步实测,8k gate_up)

| 版本 | 结构 | period | e2e |
|---|---|---|---|
| 会话起点 | f32 逐元素 transform,s3 均匀 | ~1200 | 63.5 µs |
| V3 op-count 手术 | 打包 bf16x2 转换、树 amax、整数 s0、指数 bump | 890 | 51.3 |
| R33-34 | bk64×s6(环/深度)+ 1/4-atom slot | 834 | 49.0 |
| R35 异深 | A×3 / 产物×2 / B×4 → 逻辑 s4 | 787 | 47.7 |
| R36 A-first | `full_a` 拆分到达 | **750** | **47.2** |
| — 契约内终态 —— 以下为 wrong-result 探针阶梯(R37/38) | | | |
| probe6 砍残差 | 链 575→505 | 712 | 45.2 |
| probe8 半元素 | 链→442 | 665 | 42.9 |
| probe10 微链 | 链→335(骨架地板) | 584 | 39.0 |
| probe4 零链 | transform 计算全删 | **496** | 34.8 |

**链长-period 响应曲线**(R37/38,三点拟合并被 probe10 验证):

$$E(W) \;\approx\; 0.67\,W - 125\ \ \mathrm{ns/kb},\qquad E = \text{period} - 496$$

每砍 100ns 链只返还 ~65ns——被砍部分约 1/3 本就藏在 exec 之下。归零需 $W \le 190$,而 st.128×8 + SF 存储 + LDS/循环骨架的刚性下限 ~300ns:**契约内(transform 线程承担存储义务)idle 不可关**,这是全部 20 轮消除战役 + 探针阶梯的终审。出路只在契约外:预量化 A(落到 496 地板,−26%)或 megafusion(空窗里塞下一层)。

### 3.5 Wave 量化(makespan)

均匀 tile 的 persistent makespan $= \lceil \text{tiles}/148 \rceil \times t_{\text{tile}}$。8k gate_up:320 tiles → 2.16 波、ceil 3 → 24 个 SM 背 3 块,尾波惩罚 +39%;16k → 4.32 波/ceil 5 → +16%。DeepGEMM 细 tile(bn128)波数翻倍摊薄圆整——fused 契约学不了:BLOCK_N 决定 A 重复变换次数 $N/\text{BLOCK\_N}$,bn128 会把变换翻倍压回节拍源(实测 65.75µs);bn>256 被 TMEM 512 列挡死。修复尝试全部定界存档:尾波 K-split(R28,合并链 15µs > tile 价值)、bn128 二次核(R39,毛利 1.1 < PDL 链 3-5µs)、cluster 对拆变换(R39,毛上限 3.15µs 不抵重构)。

---

## 4. Arithmetic Intensity 与瓶颈判决

每 k-block 每 SM(bm128×bn256×bk128):

$$\mathrm{FLOPs} = 2_{\text{遍}} \times 2\cdot 128\cdot 256\cdot 128 = 16.8\ \mathrm{MFLOP},\qquad \mathrm{bytes} = \underbrace{32\mathrm{K}}_{A\ \mathrm{BF16}} + \underbrace{16\mathrm{K}}_{W} + \underbrace{2\mathrm{K}}_{SFB} = 50\ \mathrm{KB}$$

$$AI = 335\ \mathrm{FLOP/B}\ \ (\text{BF16 等效有效工作则减半为 } 168) \quad\text{vs}\quad AI_{\mathrm{ridge,B300}} \approx 245$$

名义上骑在 ridge 附近——但实测判决是**两个都不是**的第三种 regime:

- **多 wave prefill:transform 递推延迟束缚**。ncu 无一饱和(DRAM 10% / L2 13% / SM 42%);时钟锁定实验(1050MHz)把 51.3µs 分解为 ~38 SM-clock-scaled + ~13 时钟不变交付底;§3.4 的响应曲线就是这个 regime 的状态方程。
- **sub-wave(tiles<148):同步链延迟地板** ~24.6µs 平坦,每 kb ~1.0-1.2µs 的 TMA→transform→barrier→发射链;逐项记账 = 每 SM 每 kb 50KB 的 L2→SMEM 交付(128 SM × 50KB / 1.19µs ≈ 5.4 TB/s 聚合 = ncu 的 "L2 13%")——**1B(BF16)vs 2B 激活交付是 fused 设计的算法学价格**。

## 5. 终态性能(2026-08-02,单会话统一口径)

口径:dual/single = graph-replay 全调用(BF16 激活进,量化 fused);mxfp8k = 只计 gemm kernel(对 mxfp8 最优惠);mxfp8g = 全 API 调用(含 per-call SF 变换;两者都不含 bf16→fp8 激活量化)。

| gate_up (µs) | dual | single | mxfp8k | mxfp8g |
|---|---|---|---|---|
| decode 16/32/128 | **10.25 / 10.27 / 14.40** | 10.25 / 10.26 / 14.37 | 10.90 / 11.69 / 17.92 | 24.62 / 24.62 / 29.72 |
| prefill 2k/8k/16k | 18.47 / 48.33 / 62.88 | 18.33 / 43.43 / 56.76 | **17.96 / 25.44 / 35.44** | 29.00 / 37.00 / 50.64 |

| down (µs) | dual | single | mxfp8k | mxfp8g |
|---|---|---|---|---|
| decode 16/32/128 | **7.16 / 7.26 / 10.26** | 7.16 / 8.16 / 10.27 | 7.61 / 10.04 / 13.09 | 14.36 / 15.07 / 18.46 |
| prefill 2k/8k/16k | 14.35 / 26.67 / 41.49 | 12.32 / 24.59 / 36.95 | **12.86 / 17.39 / 27.33** | 18.47 / 25.80 / 36.93 |

- decode 全 6 点领先(即使对 gemm-only 口径),对 full-call 领先 44-58%;swap-AB(权重上 M 侧、token tile 32)+ bk256/s4/tw8 是 decode 默认
- prefill 2k 平手区;大 M 落后为 §4 定性的结构差距,幅度与 §3.4 响应曲线自洽
- 会话净变化:8k gate_up 63.5→47.2(−26%),decode gate_up 24.6→10.25(2.4×),精度恒定 0.9954
- 复现:`./benchmarks/run_all.sh`(GPU=/ITERS= 可覆盖)

## References

1. 本仓库:[dual_nvfp4.cuh](../include/nvfp4_gemm/transform/dual_nvfp4.cuh)(Algorithm 1/2)、[sm100_bf16_dual_nvfp4_gemm.cuh](../include/nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm.cuh)(Algorithm 3)、[sm100_bf16_dual_nvfp4_gemm.hpp](../src/kernels/sm100_bf16_dual_nvfp4_gemm.hpp)(host 启发式与 SMEM 预算)
2. 实测账本(R1-R40 全部轮次、正负结果留档):[b300-deep-opt-log.html](b300-deep-opt-log.html)
3. 时序图:[r36 稳态单拍](transform-warp-timeline-r36.drawio) / [r37 链长阶梯](transform-warp-timeline-r37.drawio) / [r38 微链](transform-warp-timeline-r38.drawio) / [split + SMEM stage](transform-warp-timeline-split.drawio) / [3-stage 重叠](transform-3stage-overlap.drawio)
4. DeepSeek-AI, *DeepGEMM*(mxfp8×mxfp4 对照实现与 UTCCP 布局约定)
5. NVIDIA, *PTX ISA*:`tcgen05.mma.kind::mxf4nvf4.block_scale`、`tcgen05.cp`、`mbarrier`、`cp.async.bulk.tensor`
6. NVIDIA, *Pretraining Large Language Models with NVFP4*, arXiv:2509.25149(E4M3 分数 scale vs MXFP4 E8M0 的动机)
7. 模板:*Kimi K3 Inference Kernel Analysis: High-Performance Kernels & MXFP4 QAT*
