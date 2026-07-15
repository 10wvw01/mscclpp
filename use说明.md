# MSCCL++ Bidirectional MemoryChannel 使用说明

本文基于 `examples/tutorials/03-memory-channel/bidir_memory_channel.cu`，说明从 Bootstrap 初始化到 Device 端使用 `put`、`get`、`signal` 等原语的关键流程。

> 示例使用 `Transport::CudaIpc`。TCP Bootstrap 只负责控制面元数据交换；真正的数据传输由 GPU 通过 CUDA IPC 映射地址直接完成。

## 1. 核心对象

每个 Rank 对应一个进程和一张 GPU：

- Rank 0 → GPU 0
- Rank 1 → GPU 1

`MemoryChannelDeviceHandle` 中最关键的内容是：

```text
src_                   本地数据缓冲区
dst_                   映射后的对端数据缓冲区
packetBuffer_          本地 packet 缓冲区
inboundToken           本地接收信号的 token
remoteInboundToken     对端 token 的映射地址
expectedInboundToken   本地 wait 的期望计数
```

## 2. Host + Device 全流程图

```mermaid
flowchart TB
    START([启动两个 Rank])

    subgraph HOST[Host 阶段：建立控制面与通道]
        H1[初始化 TcpBootstrap 和 Communicator]
        H2[建立 CudaIpc Connection]
        H3[创建并交换 Semaphore token]
        H4[注册并交换 data / packet 内存]
        H5[构造 MemoryChannel]
        H6[生成 DeviceHandle 并复制到 GPU]
        H7[barrier 后启动 CUDA Graph]

        H1 --> H2 --> H3 --> H4 --> H5 --> H6 --> H7
    end

    subgraph DEVICE[Device 阶段：执行通信原语]
        D1[relaxedSignal + relaxedWait<br/>两端进入本轮]
        D2[DeviceSyncer grid barrier]
        D3{Kernel 类型}

        P1[put：本地数据写入对端]
        P2[grid barrier]
        P3[signal + wait<br/>发布并等待 Put 完成]

        G1[get：从对端读取到本地]
        G2[Kernel / Stream 完成]

        K1[putPackets：写入对端 packet]
        K2[unpackPackets：等待 flag 并解包]

        D1 --> D2 --> D3
        D3 -->|Put| P1 --> P2 --> P3
        D3 -->|Get| G1 --> G2
        D3 -->|Put Packets| K1 --> K2
    end

    START --> H1
    H7 --> D1
```

## 3. 双 Rank 时序图

```mermaid
sequenceDiagram
    participant H0 as Rank 0 Host
    participant H1 as Rank 1 Host
    participant G0 as GPU 0
    participant G1 as GPU 1

    rect rgb(240, 246, 255)
        Note over H0,H1: Host 阶段：通过 Bootstrap 交换控制面元数据
        H0->>H1: 交换 Endpoint，建立 CudaIpc Connection
        H0->>H1: 交换 Semaphore token
        H0->>H1: 交换 data / packet CUDA IPC handle
        H0->>G0: 下发 DeviceHandle
        H1->>G1: 下发 DeviceHandle
        H0->>H1: barrier，对齐后启动 Kernel
    end

    rect rgb(245, 255, 245)
        Note over G0,G1: Device 阶段：GPU 直接访问映射后的对端内存
        G0->>G1: relaxedSignal
        G1->>G0: relaxedSignal
        Note over G0,G1: relaxedWait + grid barrier

        alt Bidir Put
            G0->>G1: put：本地切片 → 对端数据区
            G1->>G0: put：本地切片 → 对端数据区
            Note over G0,G1: grid barrier
            G0->>G1: signal：发布本端 Put 完成
            G1->>G0: signal：发布本端 Put 完成
            Note over G0,G1: wait 返回后可使用对端写入的数据
        else Bidir Get
            G0->>G1: get：读取对端切片
            G1->>G0: get：读取对端切片
            Note over G0,G1: 本地 Kernel / Stream 完成表示 Get 完成
        else Bidir Put Packets
            G0->>G1: putPackets：payload + flag
            G1->>G0: putPackets：payload + flag
            Note over G0,G1: unpackPackets 轮询本地 packet flag 并解包
        end
    end
```

## 4. Host 阶段关键步骤

### 4.1 Bootstrap 与 Connection

```cpp
auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
bootstrap->initialize(ipPort);
mscclpp::Communicator comm(bootstrap);

auto conn = comm.connect(
    {mscclpp::Transport::CudaIpc,
     {mscclpp::DeviceType::GPU, gpuId}},
    remoteRank).get();
```

Bootstrap 用来交换 Endpoint 元数据，`Connection` 用于后续构造 Semaphore 和注册内存。

### 4.2 Semaphore

```cpp
auto sema = comm.buildSemaphore(conn, remoteRank).get();
```

双方各自分配一个 GPU token，并交换其 CUDA IPC 映射信息。Device 端：

```text
signal()  → 原子递增 remoteInboundToken
wait()    → 轮询本地 inboundToken
```

### 4.3 内存注册与交换

```cpp
auto localRegMem =
    comm.registerMemory(buffer.data(), buffer.bytes(), transport);

comm.sendMemory(localRegMem, remoteRank);
auto remoteRegMem = comm.recvMemory(remoteRank).get();
```

`remoteRegMem.data()` 是映射后的对端 GPU 地址，可被 Device 端直接 load/store。

### 4.4 MemoryChannel

```cpp
mscclpp::MemoryChannel memChan(
    sema,
    /* dst */ remoteRegMem,
    /* src */ localRegMem);
```

因此：

```text
put：src_ 本地内存 → dst_ 对端内存
get：dst_ 对端内存 → src_ 本地内存
```

## 5. Device 原语语义

| 原语 | 作用 | 同步语义 |
|---|---|---|
| `relaxedSignal()` | 对端 token 加 1 | 仅用于执行握手，不保证此前数据完成 |
| `relaxedWait()` | 等待本地 token | Relaxed，不提供数据可见性保证 |
| `signal()` | 发布本端此前写操作完成 | System-scope release |
| `wait()` | 等待对端发布完成 | System-scope acquire |
| `put()` | 本地内存写到对端内存 | 多线程协作 GPU copy |
| `get()` | 从对端内存读取到本地 | 多线程协作 GPU copy |
| `putPackets()` | 将 payload 和 flag 写入对端 packet | flag 表示包就绪 |
| `unpackPackets()` | 等待本地 packet flag 后解包 | 包级细粒度同步 |
| `devSyncer.sync()` | 同步 Kernel 内所有 block | 防止部分线程尚未完成就发送 signal |

## 6. 三种 Kernel 的关键差异

### Bidir Put

```text
开头握手
→ 本地数据写入对端
→ grid barrier
→ signal 发布完成
→ wait 等待对端完成
```

Put 末尾的 `devSyncer.sync()` 很重要，否则 tid 0 可能在其他线程尚未完成写入时提前执行 `signal()`。

### Bidir Get

```text
开头握手
→ 主动读取对端数据
→ Kernel / Stream 完成
```

示例中 Get 没有末尾 `signal/wait`，因为只关心本地读取何时完成。

### Bidir Put Packets

```text
开头握手
→ putPackets 写 payload + flag 到对端
→ unpackPackets 等待本地 packet 的 flag
→ 解包到本地数据区
```

Packet 模式使用包内 flag 表示数据就绪，因此不再依赖末尾的 channel semaphore。

## 7. 需要注意的点

1. `Transport::CudaIpc` 要求两个 GPU 处于可建立 CUDA IPC/P2P 映射的环境中。
2. Bootstrap 是控制面，不承载 `put/get` 的数据传输。
3. `relaxedSignal/relaxedWait` 不能替代数据完成同步。
4. `signal/wait` 应与正确的 grid/block 同步配合使用。
5. `devSyncer.sync()` 是软件 grid barrier，需要保证所有 block 能够并发驻留，避免等待未被调度的 block。
