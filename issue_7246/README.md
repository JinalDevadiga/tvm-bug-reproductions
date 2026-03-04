# Issue #7246 — Race Condition in `tvm.tir.call_packed()` Under Parallel Schedule

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/7246
- **Fix PR:** https://github.com/apache/tvm/pull/7619 (merged Apr 12, 2021)
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Closed (fixed in TVM 0.9.0)

## Approach

`apache-tvm==0.8.0` (the buggy version) is not available as a pip wheel —
the oldest available release is `0.9.0`, which already contains the fix.
We therefore use the **IR inspection approach** (same as Triton #8311):
run on `0.11.1`, dump the lowered TIR, confirm the fix is present, and
document exactly what the buggy IR looked like from the issue report and
PR diff.

## What is the Bug?

`LowerTVMBuiltin` (the TIR pass that lowers packed function calls) allocated
the packed-func argument stack — `tvm_stack_alloca` for `stack_value` and
`stack_tcode` — **once at the top of the generated function**, outside any
loop. When the enclosing loop is marked `parallel()`, all OS threads share
**the same stack memory**. Each thread reads and writes the same
`stack_value` / `stack_tcode` arrays concurrently:

```
// Buggy TIR (TVM <0.9.0)
stack_value = tvm_stack_alloca("arg_value", 8)   // <- allocated ONCE, shared
stack_tcode = tvm_stack_alloca("arg_tcode", 8)   // <- allocated ONCE, shared

parallel for (xo, 0, 2) {
  for (yo, 0, 2) {
    // Thread 0 and Thread 1 both write stack_value[0] = &A_tile, etc.
    // Last writer wins -> wrong matmul args -> wrong result or crash
    tvm_call_packed_lowered("tvm.contrib.cblas.matmul",
                            stack_value, stack_tcode, 0, 5)
  }
}
```

## The Fix (PR #7619)

The fix moves stack allocation **inside** the parallel loop so each thread
gets its own private copy:

```
// Fixed TIR (TVM 0.9.0+, PR #7619)
parallel for (xo, 0, 2) {
  stack_value = tvm_stack_alloca("arg_value", 8)   // <- thread-private
  stack_tcode = tvm_stack_alloca("arg_tcode", 8)   // <- thread-private
  for (yo, 0, 2) {
    tvm_call_packed_lowered("tvm.contrib.cblas.matmul",
                            stack_value, stack_tcode, 0, 5)
  }
}
```

## Observed TIR on TVM 0.11.1

TVM 0.11.1 goes further than the PR #7619 fix — it eliminates the stack
allocation entirely, inlining all arguments as `tvm_stack_make_array` calls
constructed **inline inside the parallel loop**:

```
for (i.outer: int32, 0, 2) "parallel" {
  for (j.outer: int32, 0, 2) {
    @tir.tvm_call_packed("tvm.contrib.cblas.matmul",
      @tir.tvm_stack_make_array(A_2, ..., cse_var_1, ...),   // <- inline, thread-safe
      @tir.tvm_stack_make_array(B_2, ..., cse_var_2, ...),   // <- inline, thread-safe
      @tir.tvm_stack_make_array(C_2, ..., ...),
      False, False, dtype=int32)
  }
}
```

No `tvm_stack_alloca` appears anywhere — arguments are constructed per-call
rather than pre-allocated on a shared stack. This is strictly safer than the
PR #7619 fix.

## Requirements

- Python 3.9 – 3.11
- `apache-tvm==0.11.1`
- No GPU or BLAS library required

## Setup

```bash
conda activate tvm-bugs    # apache-tvm 0.11.1
```

## How to Run

```bash
python reproduce.py
```

## Root Cause

`src/tir/transforms/lower_tvm_builtin.cc` — the `LowerTVMBuiltin` pass
lifted all `tvm_stack_alloca` nodes to the outermost scope (function body)
regardless of whether they were inside a parallel loop. The fix: detect
`ForKind::kParallel` and scope stack allocation inside the loop body.