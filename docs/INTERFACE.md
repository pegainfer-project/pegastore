# Interface Design v0

面向 AI 场景的 immutable large-object cache：多 slot value、slot 级 NUMA/设备位置、GPU/DRAM/SSD 多位置、跨节点全共享。

## 0. Non-Goals（每个 PR 先过这一关）

- 不是 parameter server：对象 **immutable**，`put` 是 **write-once**，没有 update/upsert/append。
- 不做 lease / pin：`get` 可能返回 `Evicted`，上层处理。store 内部有 transfer 期间的 refcount，但不暴露。
- 不做迁移：slot 落在哪就在哪，只有驱逐和重放。
- 不做配额、不做多租户、不做鉴权。namespace 是 key 前缀，由用户编码。
- 不做 durability：所有 tier（含 SSD）都是 cache，重启可清。
- 不做版本/ref：版本切换是用户控制面的事。
- 不做 LPM / 前缀匹配：store 只认精确 key，前缀索引在上层。
- 不做 partial-prefix / paged KV 语义：slot 是唯一的子对象粒度。

## 1. 概念

| 概念 | 含义 |
|---|---|
| **Key** | 不透明 bytes。索引前缀有序，支持 `remove_prefix`。用户自己编码 namespace / 内容 hash。 |
| **Object** | 一个 key 下的 slot 集合。创建时 slot 数量与长度定死；只承担命名与 spec。 |
| **Slot** | 放置、元数据、可见性、驱逐的最小单位。每个 slot 有自己的 placement。部分驻留是常态。 |
| **Replica** | slot 的一份物理副本：`(node, device, tier)` + `misplaced`。 |
| **Device** | 位置坐标：`Cpu{numa}` 或 `Gpu{index}`。距离函数由后端提供。 |
| **Tier** | `Dram` / `Ssd`（store 拥有）/ `External`（用户出借，session 生命周期）。 |
| **MemoryRegion** | 用户注册的内存。所有数据进出都通过指向它的 `Iov`。没有非注册慢路径。 |

## 2. 核心类型

```rust
use bytes::Bytes;

pub struct Key(pub Bytes);
pub struct SlotIdx(pub u32);
pub struct NodeId(pub u64);

pub enum Device { Cpu { numa: u16 }, Gpu { index: u16 } }
pub enum Tier   { Dram, Ssd, External }
pub struct Location { pub node: NodeId, pub device: Device, pub tier: Tier }

pub enum Placement {
    Strict(Device),        // 放不下 → NoSpace
    Prefer(Device),        // 放不下允许溢出到其它 device，replica.misplaced = true
    Each(Vec<Device>),     // 列出的每个 device 各一份
    Anywhere,              // 随缘
}

pub enum Retention {
    Cache,                 // 可被驱逐
    Explicit,              // 只被 remove 删除；池满时对 Cache 驱逐、对 Explicit 返回 NoSpace
}

pub struct SlotSpec   { pub len: u64, pub placement: Placement }
pub struct ObjectSpec { pub retention: Retention, pub slots: Vec<SlotSpec> }

/// 由后端 `register` 返回；Drop 时注销。
pub struct MemoryRegion { /* ptr, len, device, 后端私有句柄(rkey / cuda ipc / shm) */ }
impl MemoryRegion { pub fn device(&self) -> Device; pub fn len(&self) -> u64; }

pub struct Iov<'a> { pub region: &'a MemoryRegion, pub offset: u64, pub len: u64 }

pub struct Replica    { pub location: Location, pub misplaced: bool }
pub struct SlotInfo   { pub len: u64, pub replicas: Vec<Replica> }   // 空 = 未落地 / 已驱逐
pub struct ObjectInfo { pub retention: Retention, pub slots: Vec<SlotInfo> }
```

## 3. 操作参数（`Op*` 是数据；新需求加字段，不加方法）

```rust
pub struct PutSlot<'a> { pub spec: SlotSpec, pub src: &'a [Iov<'a>] }   // src 总长 == spec.len
pub struct OpPut<'a>   { pub retention: Retention, pub slots: Vec<PutSlot<'a>> }
pub struct RpPut       { pub slots: Vec<Result<()>> }                  // 逐 slot：Ok / AlreadyExists / NoSpace

pub struct OpGet<'a>   { pub slot: SlotIdx, pub src_offset: u64, pub dst: &'a [Iov<'a>] }
pub struct RpGet       { pub from: Location }                          // 实际读源，供 metrics / 调试

pub struct OpPublish<'a> { pub slot: SlotIdx, pub src: &'a [Iov<'a>] } // 就地作为 External replica
```

## 4. 后端 trait（实现者面对的完整接口，7 个操作 + register + info）

```rust
pub trait Access: Send + Sync + 'static {
    fn info(&self) -> Arc<AccessInfo>;
    fn register(&self, ptr: NonNull<u8>, len: u64, device: Device) -> Result<MemoryRegion>;

    fn put<'a>(&'a self, key: Key, op: OpPut<'a>)
        -> impl Future<Output = Result<RpPut>> + Send + 'a;
    fn publish<'a>(&'a self, key: Key, op: OpPublish<'a>)
        -> impl Future<Output = Result<()>> + Send + 'a;
    fn stat<'a>(&'a self, keys: &'a [Key])
        -> impl Future<Output = Result<Vec<Option<ObjectInfo>>>> + Send + 'a;
    fn get<'a>(&'a self, key: Key, op: OpGet<'a>)
        -> impl Future<Output = Result<RpGet>> + Send + 'a;
    fn remove<'a>(&'a self, keys: &'a [Key])
        -> impl Future<Output = Result<()>> + Send + 'a;
    fn remove_prefix<'a>(&'a self, prefix: &'a [u8])
        -> impl Future<Output = Result<u64>> + Send + 'a;
}

/// dyn 镜像：`*_dyn` 返回 BoxedFuture；blanket impl<T: Access> AccessDyn for T；
/// impl Access for Arc<dyn AccessDyn>。Python binding 与 Layer 持有 `Servicer = Arc<dyn AccessDyn>`。
pub trait AccessDyn: Send + Sync + 'static { /* 同名 *_dyn 方法 */ }
pub type Servicer = Arc<dyn AccessDyn>;

pub struct AccessInfo {
    pub name: &'static str,          // "memory" | "local" | "remote"
    pub node: NodeId,
    pub devices: Vec<Device>,        // 本后端可见的 device
    pub capability: Capability,
}
```

## 5. 能力声明（每个可选行为一个 bool，没有隐式 fallback）

```rust
pub struct Capability {
    pub gpu_memory: bool,            // register(Device::Gpu) 可用
    pub publish: bool,
    pub placement_strict: bool,
    pub placement_each: bool,
    pub retention_explicit: bool,
    pub remove_prefix: bool,
    pub max_slot_len:   Option<u64>,
    pub max_slots:      Option<u32>,
    pub max_iov:        Option<u32>, // 单次 src/dst 段数上限
    pub max_stat_batch: Option<u32>,
}
```

`Store` 层在进后端前检查，不支持直接返回 `Unsupported`。

## 6. 错误（给程序读：kind 闭集 + 可重试性 + context）

```rust
pub enum ErrorKind {
    NotFound,        // key 不存在                       Permanent
    Evicted,         // slot 存在过但已无 replica         Permanent（重算/重放）
    AlreadyExists,   // write-once 冲突                  Permanent
    SpecMismatch,    // 同 key 不同 spec                  Permanent
    NoSpace,         // Strict 放不下 / Explicit 池满     Temporary（背压）
    Unsupported,     // Capability 说不行                 Permanent
    InvalidInput,    // iov 长度、offset、region device 不匹配 Permanent
    Unavailable,     // daemon / metaserver / 远端不可达   Temporary
    Unexpected,
}

pub enum ErrorStatus { Permanent, Temporary }

pub struct Error {
    kind: ErrorKind, status: ErrorStatus,
    operation: &'static str,
    context: Vec<(&'static str, String)>,      // key / slot / device / node
    source: Option<anyhow::Error>,
}
impl Error {
    pub fn kind(&self) -> ErrorKind;
    pub fn is_temporary(&self) -> bool;
    pub fn with_operation(self, op: &'static str) -> Self;
    pub fn with_context(self, k: &'static str, v: impl ToString) -> Self;
}
```

**`Unavailable` 永远不能被当成 miss。** RetryLayer 只重试 `Temporary`。

## 7. 用户层（三形态：简、`_with` builder、`_options`）

```rust
#[derive(Clone)]
pub struct Store { srv: Servicer, info: Arc<AccessInfo> }

impl Store {
    pub fn new(acc: impl Access) -> Self;
    pub fn layer<L: Layer>(self, l: L) -> Self;
    pub fn info(&self) -> &AccessInfo;
    pub fn register(&self, ptr: NonNull<u8>, len: u64, device: Device) -> Result<MemoryRegion>;

    // put
    pub async fn put(&self, key: Key, slots: Vec<PutSlot<'_>>) -> Result<RpPut>;                  // Retention::Cache
    pub fn put_with(&self, key: Key, slots: Vec<PutSlot<'_>>) -> FuturePut<'_>;                    // .retention(Explicit).await
    pub async fn put_options(&self, key: Key, op: OpPut<'_>) -> Result<RpPut>;

    // get
    pub async fn get(&self, key: Key, slot: SlotIdx, dst: &[Iov<'_>]) -> Result<RpGet>;           // src_offset = 0
    pub fn get_with(&self, key: Key, slot: SlotIdx, dst: &[Iov<'_>]) -> FutureGet<'_>;             // .src_offset(n).await
    pub async fn get_options(&self, key: Key, op: OpGet<'_>) -> Result<RpGet>;
    /// 按完成顺序流式返回；layer-wise overlap 的载体。Store 层用 FuturesUnordered 拼，不进 Access。
    pub fn get_many<'a>(&'a self, reqs: Vec<(Key, OpGet<'a>)>)
        -> impl Stream<Item = (usize, Result<RpGet>)> + 'a;

    pub async fn publish(&self, key: Key, slot: SlotIdx, src: &[Iov<'_>]) -> Result<()>;
    pub async fn stat(&self, keys: &[Key]) -> Result<Vec<Option<ObjectInfo>>>;
    pub async fn remove(&self, keys: &[Key]) -> Result<()>;
    pub async fn remove_prefix(&self, prefix: &[u8]) -> Result<u64>;
}

// builder = OperatorFuture 模式：持 (srv, key, args, fn)，setter 改 args 字段，impl IntoFuture。
pub struct FuturePut<'a> { /* ... */ }  impl FuturePut<'_> { pub fn retention(self, r: Retention) -> Self; }
pub struct FutureGet<'a> { /* ... */ }  impl FutureGet<'_> { pub fn src_offset(self, n: u64) -> Self; }
```

## 8. 语义

**put**
1. key 不存在：原子创建 metadata（spec 落地，所有 slot 为空），随后逐 slot 写入。
2. key 已存在：spec 必须一致（否则 `SpecMismatch`）；仅写当前**无 replica 且无进行中写入**的 slot，其余返回 `AlreadyExists`。
3. slot 数据落地即可见。put 中途失败留下空 slot，可被后续 put 幂等重填。
4. 两个客户端同时 put 同一 key：后到者在 slot 级拿到 `AlreadyExists`，可据此跳过 D2H。
5. `Placement` 逐 slot 生效；`Prefer` 溢出的 replica 标 `misplaced`，永不迁移。

**get**
1. 源选择：`distance(dst.device, replica.location)` 最小者；同节点 GPU→GPU 走 NVLink，GPU↔DRAM 走 C2C/PCIe，跨节点 RDMA 双边选贴近 src/dst device 的 NIC；某 NUMA 无 NIC 时用任意 NIC 直接 DMA，不 bounce。
2. 无 replica → `Evicted`；key 不存在 → `NotFound`。
3. 传输期间 store 内部对源 replica 持 refcount，防止读到被复用的内存。
4. `dst` 总长必须等于 `min(slot.len - src_offset, ...)`，否则 `InvalidInput`。

**publish**
- 把用户已注册内存登记为某 slot 的 `External` replica，不拷贝。生命周期 = 本 session；session 断开即从 metadata 消失。用于 peer-assisted read（GPU0 get 后 publish 给 NVLink 邻居 / 其它节点）与零拷贝发布已在显存的权重。
- 目标 slot 不存在 → `NotFound`；`src` 长度 ≠ slot.len → `InvalidInput`。

**驱逐**
- 每个 `(node, device, tier)` 池独立驱逐；DRAM 与 SSD 均按 extent（log-structured）整体回收；SSD 按 NUMA 分组。
- 只驱逐 `Retention::Cache`；`Explicit` 只能被 `remove`。Explicit 占满池 → 新 put 返回 `NoSpace`，这就是背压。
- 驱逐是 slot 级；object 在所有 slot 都空且无引用后由后台回收 metadata。

**remove / remove_prefix**
- 立即从 metadata 摘除；物理空间在进行中的 get 完成后回收。
- `remove_prefix` 依赖前缀有序索引；返回摘除的 object 数。

## 9. Layer（核心保持裸）

```rust
pub trait Layer: Send + Sync + 'static {
    fn layer(&self, inner: Servicer) -> Servicer;
}
```
首批：`RetryLayer`（仅 `Temporary`，指数退避）、`TimeoutLayer`、`MetricsLayer`（按 operation × kind × from.tier 计数/直方图）、`TracingLayer`、`ConcurrencyLimitLayer`（per-device 在途上限）。

## 10. 后端与测试

| 后端 | 用途 |
|---|---|
| `Memory` | 语义参考实现；单进程；所有 Capability 为 true。行为测试的 oracle。 |
| `Local` | 连本节点 daemon：UDS 控制面 + shm / CUDA IPC 数据面。daemon 拥有 per-NUMA DRAM extent 池与 per-NUMA SSD ring。 |
| `Remote` | metaserver 目录 + RDMA one-sided READ；也承担 `Local` 的跨节点 miss 路径。 |

行为测试一套跑三后端；RDMA 后端与 Memory 行为不一致即 bug。

## 11. 三个 workload

- **权重**：`put_with(key, slots).retention(Explicit)`，`Placement::Each([Cpu0, Cpu1])` 或 `Prefer(Cpu_of_gpu)` + GPU0 `get` 后 `publish` 给邻居走 NVLink。一批 tensor 各一个 object，manifest 与版本切换在用户控制面。
- **KV checkpoint**：`put(Cache)`，slot = TP rank（或 rank × layer 组），`Prefer(Cpu_of_producing_gpu)`。decode 侧 `stat` 拿位图 → `get_many` 流式收。LPM 索引在上层。
- **hidden state（MTP 训练）**：`put(Explicit)`，池满 `NoSpace` 即背压；trainer `stat → get → remove`。

## 12. 明确推迟（不是不做，是先不做）

- `prefetch(keys, to: Device)`：是用户驱动的迁移；等有跨节点 miss 数据再加。加法形式：`Access` 新方法 + Capability 一个 bool。
- 小对象 inline 路径（≤ 64 KiB 走 RPC payload 不注册内存）。
- `OpGet.completion`：CUDA event / shm 状态位替代 future 轮询。加字段即可，不破坏接口。
- `Placement` 的用户自定义策略钩子（"随缘"之外的第五种）。
