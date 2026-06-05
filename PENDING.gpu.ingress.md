# GPU-Native Dispatch-Kind Ingress Plan

## Diagnosis

Phase 9 is structurally complete, but ingress classification is incomplete.

The runtime already has:

- GPU scheduler dispatch-kind support via `TensorQuantaleWorld::set_dispatch_kinds`.
- GPU-to-host command transport via `DeviceCommand`.
- Host-to-GPU external receipts via `DeviceReceiptExt`.
- Host service handling via `orch_service::service_external_commands`.

The missing bridge is a default node-level dispatch-kind compiler and upload
step:

\[
T \xrightarrow{\text{classify}} K \xrightarrow{\text{upload}} G
\]

where:

\[
G=\text{GPU scheduler}
\]

\[
H=\text{host / CPU service layer}
\]

\[
N=\text{topology nodes}
\]

\[
K:N\rightarrow D
\]

\[
D=\{\text{HF_DEVICE},\text{ABSTRACT_DEVICE},\text{EXTERNAL_PROCESS},\text{EXTERNAL_IO},\text{UNSUPPORTED}\}
\]

\[
C=\text{DeviceCommand ring}
\]

\[
R=\text{DeviceReceiptExt ring}
\]

\[
T=\text{topology/operator metadata}
\]

\[
Q=\text{quantale tensor runtime}
\]

The default CUDA runtime must make old-world host work visible to the GPU
scheduler as external work. It should not reintroduce CPU scheduling fallback.

## Core Equation

\[
\boxed{
T \xrightarrow{\text{classify}} K \xrightarrow{\text{upload}} G
}
\]

\[
\boxed{
G(n)=
\begin{cases}
\text{execute_device}(n), & K(n)\in\{\text{HF_DEVICE},\text{ABSTRACT_DEVICE}\}\\
\text{emit}(C,n), & K(n)\in\{\text{EXTERNAL_PROCESS},\text{EXTERNAL_IO}\}\\
\text{block/repair}(n), & K(n)=\text{UNSUPPORTED}
\end{cases}
}
\]

One-line explanation:

\[
\text{The missing bridge is a default node-level dispatch-kind compiler and upload step.}
\]

## Proposed Solution

### 1. Add Official Default Dispatch-Kind Construction

Create a canonical function:

```rust
fn build_node_dispatch_kinds(
    topology: &GraphTopology,
    config: &SystemConfig,
) -> Vec<i32>
```

It should return exactly:

```rust
Vec<i32> // length = TENSOR_NODE_COUNT
```

Classification rule:

```text
if node is hot/fusion/HF covered:
    DISPATCH_KIND_HF_DEVICE

else if node is abstract-device covered:
    DISPATCH_KIND_ABSTRACT_DEVICE

else if operator kind == python/process/llm:
    DISPATCH_KIND_EXTERNAL_PROCESS

else if operator kind == io/market_feed/file/network:
    DISPATCH_KIND_EXTERNAL_IO

else if node is Control::Halt or a pure control/event node handled by the device scheduler:
    DISPATCH_KIND_HF_DEVICE

else:
    DISPATCH_KIND_UNSUPPORTED
```

Then in the default CUDA runtime path, before:

```rust
gpu_native_supervisor_loop(...)
```

call:

```rust
let dispatch_kinds = build_node_dispatch_kinds(&epoch.topology.document, &config);
epoch.world.set_dispatch_kinds(&dispatch_kinds)?;
```

This is the main fix.

### 2. Source Of Truth Should Be Operator Metadata

Use `config.operator_registry` loaded from `assets/operators.generated.json` or
`assets/operators.json`. Avoid fragile node-name guessing except as a final
fallback.

Expected metadata categories:

```json
{
  "State::MarketFeed": {
    "kind": "external_io"
  },
  "State::AnalysisPlan": {
    "kind": "external_process"
  },
  "Action::CallLLM": {
    "kind": "external_process"
  },
  "Execution::VectorAdd": {
    "kind": "device"
  },
  "Execution::VectorScale": {
    "kind": "device"
  }
}
```

If the existing schema has another field like `runtime`, `executor`, `path`,
`implementation`, or `executable`, map from that.

Canonical mapping:

\[
\text{python file} \Rightarrow \text{EXTERNAL_PROCESS}
\]

\[
\text{market/feed/io/file/network} \Rightarrow \text{EXTERNAL_IO}
\]

\[
\text{fusion/hot/hf/abstract-device} \Rightarrow \text{GPU-native}
\]

Current metadata already indicates relevant executable categories:

- `State::MarketFeed` uses `python3` with `market_feed.py`.
- `State::AnalysisPlan` uses `python3` with `call_llm.py --template analysis`.
- `State::TradePlan` uses `python3` with `call_llm.py --template trade`.
- JIT/hot analysis and execution nodes are represented by hot/fusion/HF metadata.

### 3. Add Runtime Invariant Tests

Add a CUDA test:

```rust
#[test]
fn default_runtime_uploads_external_dispatch_kinds() {
    let kinds = build_node_dispatch_kinds(&topology, &config);

    assert_eq!(
        kinds[id("State::MarketFeed")],
        DISPATCH_KIND_EXTERNAL_IO
    );

    assert_eq!(
        kinds[id("State::AnalysisPlan")],
        DISPATCH_KIND_EXTERNAL_PROCESS
    );
}
```

Then add an end-to-end test:

```rust
#[test]
fn gpu_native_reaches_external_command_from_default_runtime() {
    let mut epoch = RuntimeEpoch::build_default_cuda(...)?;
    let status = epoch.world.orchestrate_until_wait_or_halt(64)?;

    assert_eq!(status, OrchStepStatus::WaitExternal);

    let cmds = epoch.world.drain_device_commands()?;
    assert!(cmds.iter().any(|c| {
        c.dispatch_kind == DISPATCH_KIND_EXTERNAL_PROCESS
            || c.dispatch_kind == DISPATCH_KIND_EXTERNAL_IO
    }));
}
```

Reason:

\[
\text{manual test passes} \not\Rightarrow \text{default runtime uploads table}
\]

The lower-level mechanism exists. The missing proof is default integration.

### 4. Do Not Resurrect The Old CPU Loop

Do not route market data or LLM calls through the legacy CPU scheduler.

Correct direction:

\[
\boxed{
CPU\neq scheduler
}
\]

\[
\boxed{
CPU=external\ service(C,R)
}
\]

Runtime flow:

```text
GPU selects State::MarketFeed
GPU sees DISPATCH_KIND_EXTERNAL_IO
GPU emits DeviceCommand
CPU service drains command
CPU runs market_feed.py
CPU pushes DeviceReceiptExt
GPU drains receipt
GPU continues
```

This preserves the Phase 9 direction.

### 5. Fix The `Execution::VectorScale` Topology Warning Separately

Current warning:

```text
Execution::VectorScale has 1 outgoing edge but 2 incoming edges
single exit is consumed on first traversal
re-entry via different predecessor blocks until hard reset
```

Math:

\[
in(n)>1 \land out(n)=1 \land consumed(exit(n))=1 \Rightarrow reentry(n)=blocked
\]

Best fix:

```text
Execution::VectorScale::FromVectorAdd
Execution::VectorScale::FromOtherPath
```

Each gets one incoming path:

\[
in(n_i)=1
\]

Alternative fix:

```json
{
  "node": "Execution::VectorScale",
  "consumption": "reentrant"
}
```

Only do this if repeated traversal is safe.

Avoid hard-resetting around it as normal behavior. That hides a graph-shape bug.

## Implementation Order

### Step 1

Add:

```rust
build_node_dispatch_kinds(...)
```

Preferred location:

```text
src/dispatch_kind.rs
```

This is now a canonical compiler layer, not a runtime-loop detail.

### Step 2

Call it in the CUDA-native path before supervisor start:

```rust
let dispatch_kinds = build_node_dispatch_kinds(&epoch.topology.document, &config);
epoch.world.set_dispatch_kinds(&dispatch_kinds)?;
```

This should happen after topology/operator/fusion/abstract-device registries are
loaded, before:

```rust
gpu_native_supervisor_loop(...)
```

### Step 3

Log the table summary once:

```text
[gpu_native] [INFO] dispatch_kinds_uploaded hf_device=N abstract_device=N external_process=N external_io=N unsupported=N
```

Expected target:

```text
external_process > 0
external_io > 0
```

If both are zero, old LLM/market nodes are still invisible to GPU-native
orchestration.

### Step 4

Add acceptance tests:

```bash
rtk cargo test --features cuda dispatch_kind
rtk cargo test --features cuda scheduler_emits_external_command
rtk cargo run --features cuda
```

Target log should eventually show:

```text
[gpu_native] [INFO] burst_complete status=WaitExternal ...
[gpu_native] [INFO] external_commands_serviced count=1
```

Then later:

```text
[gpu_native] [INFO] halted ...
```

## Codex Instruction

```text
Implement official GPU-native dispatch-kind ingress for quantale_semiring_v2.

Repository:
neurosymbolic/quantale_semiring_v2

Problem:
The CUDA scheduler supports node-level dispatch kinds through
TensorQuantaleWorld::set_dispatch_kinds, DeviceCommand, DeviceReceiptExt, and
orch_service::service_external_commands. Tests manually set dispatch kinds and
prove external commands work. But the default runtime does not appear to
build/upload a node-level dispatch-kind table from topology/operator metadata,
so market feed and LLM nodes are not reliably emitted as DeviceCommands in the
GPU-native supervisor path.

Goal:
Add a canonical default dispatch-kind compiler and upload it before
gpu_native_supervisor_loop.

Requirements:
1. Add a function that builds Vec<i32> of length TENSOR_NODE_COUNT.
2. Classify each topology node:
   - HF/fusion/hot-region covered => DISPATCH_KIND_HF_DEVICE
   - abstract-device covered => DISPATCH_KIND_ABSTRACT_DEVICE
   - Python/process/LLM operator => DISPATCH_KIND_EXTERNAL_PROCESS
   - IO/market-feed/file/network operator => DISPATCH_KIND_EXTERNAL_IO
   - Control::Halt and pure control nodes that device scheduler handles => DISPATCH_KIND_HF_DEVICE
   - unknown unsupported nodes => DISPATCH_KIND_UNSUPPORTED
3. Use assets/operators.json or existing operator registry metadata as the source of truth.
   Avoid fragile node-name guessing except as a final fallback.
4. In the default CUDA runtime path, call epoch.world.set_dispatch_kinds(&kinds)?
   before gpu_native_supervisor_loop.
5. Add one summary log:
   [gpu_native] [INFO] dispatch_kinds_uploaded hf_device=N abstract_device=N external_process=N external_io=N unsupported=N
6. Add tests:
   - default dispatch table marks market feed as EXTERNAL_IO.
   - default dispatch table marks LLM/process Python nodes as EXTERNAL_PROCESS.
   - default CUDA runtime can reach ORCH_WAIT_EXTERNAL and drain at least one DeviceCommand
     when a reachable external node exists.
7. Do not re-enable or route through legacy CPU orchestration.
8. Keep CPU role limited to external command service and receipt pushback.
9. Also add or update a topology invariant test for consumed block points.
   Execution::VectorScale currently has two incoming edges and one outgoing
   consumed edge; either split the node by predecessor or mark it explicitly
   reentrant only if safe.

Verification:
- rtk cargo check --features cuda
- rtk cargo check --features legacy-cpu-orchestration,cuda
- rtk cargo test --lib --tests
- rtk cargo test --features cuda
- rtk cargo run --features cuda

Expected runtime evidence:
- dispatch_kinds_uploaded shows external_process > 0 or external_io > 0.
- GPU-native run reaches WaitExternal when external nodes are reachable.
- orch_service services at least one command.
- GPU drains DeviceReceiptExt and continues.
- Normal halt still works.
```

## Final Diagnosis

\[
\boxed{
\text{Phase 9 is structurally complete, but ingress classification is incomplete.}
}
\]

The system has the GPU command/receipt protocol. The missing piece is making
old-world work visible to the GPU scheduler as external work.

\[
\boxed{
\text{Fix } T\rightarrow K\rightarrow G,\text{ not } G\rightarrow H\text{ scheduler fallback.}
}
\]

## Implementation Status

Status: implemented.

Landed:

- Added canonical node-level dispatch-kind compilation in `src/dispatch_kind.rs`.
- Exported dispatch-kind ingress helpers from `src/lib.rs`.
- Uploaded the default dispatch-kind table in the CUDA-native runtime before
  `gpu_native_supervisor_loop`.
- Added the `dispatch_kinds_uploaded` summary log with counts for HF device,
  abstract device, external process, external IO, and unsupported nodes.
- Kept legacy CPU orchestration disabled by default; host/CPU remains the
  external command service and receipt pushback layer.
- Added focused tests proving default metadata marks `State::MarketFeed` as
  `EXTERNAL_IO`, Python/process plan nodes as `EXTERNAL_PROCESS`, and a
  reachable external IO node yields `ORCH_WAIT_EXTERNAL` plus a `DeviceCommand`.
- Updated supervisor burst logging to read the actual device orchestration
  state after each burst and report `state.step`, selected node/edge, block
  state, and pending command/receipt counters.
- Added blocked/no-progress guards so the GPU-native supervisor exits cleanly
  at a known blocked topology point instead of spinning.

Verification completed:

- `rtk cargo check --features cuda`
- `rtk cargo check --features legacy-cpu-orchestration,cuda`
- `rtk cargo test --features cuda dispatch_kind -- --nocapture`
- `rtk cargo test --features cuda scheduler_emits_external_command -- --nocapture`
- `rtk cargo test --features cuda default_dispatch_table_reaches_market_feed_external_command -- --nocapture`
- `rtk cargo run --features cuda`

Runtime evidence:

```text
[gpu_native] [INFO] dispatch_kinds_uploaded hf_device=10 abstract_device=18 external_process=25 external_io=1 unsupported=7
[gpu_native] [INFO] burst_complete status=WaitExternal ...
[gpu_native] [INFO] external_commands_serviced count=1
```

Remaining separate diagnostics:

- `Execution::VectorScale` still trips the consumed-block-point topology
  warning. Fix that as a graph-shape issue by splitting the node per predecessor
  or explicitly marking it reentrant only if repeated traversal is safe.
- GPU-native receipt accounting currently reports negative
  `pending_receipt_count` while external commands are serviced. That is separate
  from ingress classification and should be debugged in the device receipt
  accounting path.
