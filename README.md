# TVM Bug Reproductions

This repository contains reproductions of data-race and concurrency bugs found
in the Apache TVM ML compiler framework, covering CPU threading races,
thread-unsafe shared state, and silent wrong-result bugs from incorrect
parallelisation.

## Bugs

| Issue | Title | Bug Type | Status |
|-------|-------|----------|--------|
| [#7246](issue_7246/) | Race condition in `tvm.tir.call_packed()` under parallel schedule | WAW race — packed-func arg stack allocated once outside parallel loop, shared by all threads | Closed |
| [#10210](issue_10210/) | Silent wrong results when `parallel()` applied to reduction axis | WAW race — multiple threads write same output accumulator without atomics | Open |
| [#17072](issue_17072/) | Race condition in TIR `ComputationCache` (CSE pass) | WAW race — static cache shared across all threads, concurrent `tvm.build()` calls race on HashTable insert/find | Closed |
| [#17439](issue_17439/) | `ThreadStorageSync` pass must run after `MergeSharedMemoryAllocations` | Missing barrier — sync inserted for separate buffers becomes misplaced after memory merge → data corruption on CUDA | Open |

## Setup

```bash
conda create -n tvm-bugs python=3.10 -y
conda activate tvm-bugs
pip install apache-tvm==0.11.1
pip install numpy
```

## Repository Structure

```
tvm-bug-reproductions/
├── README.md
├── issue_7246/
│   ├── README.md
│   └── reproduce.py
├── issue_10210/
│   ├── README.md
│   └── reproduce.py
├── issue_17072/
│   ├── README.md
│   └── reproduce.py
└── issue_17439/
    ├── README.md
    └── reproduce.py
```

## Notes

- **Issue #7246** — `apache-tvm==0.8.0` is not available as a pip wheel
  (oldest is 0.9.0). TIR inspection approach used: `reproduce.py` dumps the
  lowered TIR and confirms the buggy `tvm_stack_alloca` pattern. No BLAS
  library required.

- **Issue #10210** — CPU-only silent data race. Bug fires consistently as
  large numerical errors (~27–31× on MX450). `reproduce.py` runs 10 trials
  and reports the error range.

- **Issue #17072** — Race requires high core-count hardware to trigger
  reliably (reported on 52-core Xeon). On MX450, segfault observed at
  32 threads / 500 tasks. Source inspection of the faulty `static
  ComputationCache` declaration is the primary evidence.

- **Issue #17439** — CUDA codegen not available in the pip wheel build. TIR
  inspection approach used: `reproduce.py` applies `ThreadSync` and
  `MergeDynamicSharedMemoryAllocations` passes individually and dumps the
  TIR at each stage to show the misplaced barriers.