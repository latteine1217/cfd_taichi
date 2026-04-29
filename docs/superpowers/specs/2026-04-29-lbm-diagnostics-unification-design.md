# LBM Diagnostics Unification Design

## Goal

讓 CaseRunner 能統一驅動三條 LBM 路徑（single-phase / multiphase / Cahn-Hilliard），
並在 BenchmarkRegistry 補上 Rayleigh-Taylor 系列條目。

## Architecture

### Step Dispatch 修正

CaseRunner 目前假設所有 LBM solver 使用雙緩衝（`f_src/f_dst`）步進，
但 MultiphaseLBMSolver 和 CHLBMSolver 是自管緩衝的單參數 `step()`。

修正方式：在 solver class-level 加 `uses_dual_buffer: bool`，
CaseRunner `step_once()` 讀取此旗標決定呼叫方式。

```
CaseRunner.step_once()
  ├─ solver.solver_family == "lbm"
  │   ├─ stepper 已注入            → stepper(runner, next_step)
  │   ├─ uses_dual_buffer == True  → solver.step(f_src, f_dst)   [現有]
  │   └─ uses_dual_buffer == False → solver.step()               [新增]
  └─ else                          → solver.step()               [現有]
```

| Class | `uses_dual_buffer` |
|-------|--------------------|
| `LBMSolver` | `True` |
| `MultiphaseLBMSolver` | `False` |
| `CHLBMSolver` | `False` |

`getattr(solver, "uses_dual_buffer", True)` 確保無此 attr 的 legacy solver 不回歸。

### Diagnostics 資料流

```
CaseRunner.prepare_observables()
  └─ hasattr(solver, "prepare_diagnostics")
      ├─ True  → solver.prepare_diagnostics(f_src=None, reset_baseline=...)
      └─ False → solver.get_diagnostics()
```

`prepare_diagnostics()` 以 `f_src=None` 呼叫時，Multiphase / CH 忽略此參數並自管緩衝，
回傳的 dict 包含 `total_mass + initial_mass`（Multiphase）或 `total_phi + initial_phi`（CH），
與 CaseRunner 的 `_augment_diagnostics()` 現有的 mass_error 計算相容。

---

## Data Model

### MultiphaseLBMSolver 新增欄位

```python
# scalar ti.field（在 __init__ 配置，shape=()）
self.total_mass_a   = ti.field(dtype=ti.f32, shape=())
self.total_mass_b   = ti.field(dtype=ti.f32, shape=())
self.initial_mass_a = ti.field(dtype=ti.f32, shape=())
self.initial_mass_b = ti.field(dtype=ti.f32, shape=())
```

```python
@ti.kernel
def _compute_mass(self):
    self.total_mass_a[None] = 0.0
    self.total_mass_b[None] = 0.0
    for i, j in ti.ndrange(self.nx, self.ny):
        ig, jg = i + 1, j + 1
        if self.mask[ig, jg] == 0:
            ti.atomic_add(self.total_mass_a[None], self.rhoA[ig, jg])
            ti.atomic_add(self.total_mass_b[None], self.rhoB[ig, jg])

def prepare_diagnostics(self, f_src=None, reset_baseline: bool = False):
    self._compute_mass()
    if reset_baseline:
        self.initial_mass_a[None] = self.total_mass_a[None]
        self.initial_mass_b[None] = self.total_mass_b[None]
    u_max = float(self.get_diagnostics()["u_max"])
    return {
        "total_mass":   float(self.total_mass_a[None] + self.total_mass_b[None]),
        "initial_mass": float(self.initial_mass_a[None] + self.initial_mass_b[None]),
        "total_mass_a": float(self.total_mass_a[None]),
        "total_mass_b": float(self.total_mass_b[None]),
        "u_max":        u_max,
        "step_count":   int(self.step_count),
    }
```

### CHLBMSolver 新增欄位

```python
self.total_phi   = ti.field(dtype=ti.f32, shape=())
self.initial_phi = ti.field(dtype=ti.f32, shape=())
```

```python
@ti.kernel
def _compute_phi_sum(self):
    self.total_phi[None] = 0.0
    for i, j in ti.ndrange(self.nx, self.ny):
        ti.atomic_add(self.total_phi[None], self.phi[i + 1, j + 1])

def prepare_diagnostics(self, f_src=None, reset_baseline: bool = False):
    self._compute_phi_sum()
    if reset_baseline:
        self.initial_phi[None] = self.total_phi[None]
    phi_drift = abs(float(self.total_phi[None]) - float(self.initial_phi[None])) \
                / (self.nx * self.ny)
    return {
        "total_phi":   float(self.total_phi[None]),
        "initial_phi": float(self.initial_phi[None]),
        "phi_drift":   phi_drift,
        "u_max":       float(self.get_diagnostics()["u_max"]),
        "step_count":  int(self.step_count),
    }
```

---

## Diagnostics Helpers

### MultiphaseDiagnostics.check_convergence(tol)

語意：**發散偵測**（mass_error 超標 → 停止）

```python
def check_convergence(self, tol: float = 1e-2) -> bool:
    """
    Why: RT 模擬不收斂至穩態；mass_error 超標代表數值發散
    True = 應停止（發散），False = 繼續
    """
    if not self._history_samples:
        return False
    return bool(self._history_samples[-1].get("mass_error", 0.0) > tol)
```

### CHDiagnostics.check_convergence(tol)

語意：**相場守恆發散偵測**（phi_drift 超標 → 停止）

```python
def check_convergence(self, tol: float = 1e-2) -> bool:
    """
    Why: CH phi 總量守恆；drift 超標代表界面不穩定或數值發散
    """
    if not self._history_samples:
        return False
    return bool(self._history_samples[-1].get("phi_drift", 0.0) > tol)
```

---

## CaseRunner 修正

### `step_once()` — 加 `uses_dual_buffer` 分支

```python
elif getattr(self.solver, "uses_dual_buffer", True):
    f_src = self.solver.f if next_step % 2 == 1 else self.solver.f_new
    f_dst = self.solver.f_new if next_step % 2 == 1 else self.solver.f
    result = self.solver.step(f_src, f_dst)
else:
    result = self.solver.step()
```

### `active_distribution_field()` — 加 `uses_dual_buffer` 守衛

MultiphaseLBMSolver 無 `f / f_new`（只有 `fA / fB`），若不加守衛會 AttributeError。
`prepare_observables()` 把此方法回傳值當 `f_src` 傳給 `prepare_diagnostics()`，
Multiphase / CH 的 `prepare_diagnostics()` 忽略 `f_src`，只需確保不傳錯 field。

```python
def active_distribution_field(self):
    if self.solver is None or getattr(self.solver, "solver_family", None) != "lbm":
        return None
    if not getattr(self.solver, "uses_dual_buffer", True):
        return None   # Multiphase / CH 自管 buffer，f_src 忽略
    return self.solver.f_new if self.current_step % 2 == 1 else self.solver.f
```

---

## BenchmarkRegistry

### 新增 builder functions

兩個 RT example 補 `build_rayleigh_taylor_multiphase_runner` 和 `build_rayleigh_taylor_ch_runner`，
回傳 `tuple[CaseRunner, dict]`，參數對齊現有 `run_xxx` 函數，加 `steps` 控制快速驗收步數。

### BenchmarkSpec 接受準則

**`rayleigh_taylor_multiphase`**（tag: `lbm`, `multiphase`, `verification`）

| 準則名稱 | metric | 條件 |
|----------|--------|------|
| `history-sampled` | `history_samples` | ≥ 1 |
| `flow-active` | `u_max` | ≥ 1e-6 |
| `mass-conserved` | `mass_error` | ≤ 5e-2 |

**`rayleigh_taylor_ch`**（tag: `lbm`, `cahn_hilliard`, `verification`）

| 準則名稱 | metric | 條件 |
|----------|--------|------|
| `history-sampled` | `history_samples` | ≥ 1 |
| `flow-active` | `u_max` | ≥ 1e-6 |
| `phi-conserved` | `phi_drift` | ≤ 5e-2 |

---

## Files Modified / Created

| 動作 | 路徑 | 內容 |
|------|------|------|
| Modify | `src/lbm_taichi/core/multiphase_solver.py` | `uses_dual_buffer`, scalar fields, `_compute_mass`, `prepare_diagnostics` |
| Modify | `src/lbm_taichi/core/ch_lbm_solver.py` | `uses_dual_buffer`, scalar fields, `_compute_phi_sum`, `prepare_diagnostics` |
| Modify | `src/lbm_taichi/core/lbm_solver.py` | `uses_dual_buffer = True` |
| Modify | `src/lbm_taichi/core/multiphase_diagnostics.py` | `check_convergence(tol)` |
| Modify | `src/lbm_taichi/core/ch_diagnostics.py` | `check_convergence(tol)` |
| Modify | `src/cfd_taichi/case_runner.py` | `step_once()` 加 `uses_dual_buffer` 分支；`active_distribution_field()` 加守衛 |
| Modify | `examples/rayleigh_taylor_multiphase.py` | 加 `build_rayleigh_taylor_multiphase_runner` |
| Modify | `examples/rayleigh_taylor_ch.py` | 加 `build_rayleigh_taylor_ch_runner` |
| Modify | `src/cfd_taichi/benchmark_registry.py` | 兩個 RT BenchmarkSpec 條目 |
| Create | `tests/test_lbm_diagnostics_unification.py` | 整合測試 |

## Testing Strategy

`tests/test_lbm_diagnostics_unification.py`：

1. `test_multiphase_step_dispatch_via_case_runner` — CaseRunner 能 `step_once()` MultiphaseLBMSolver（驗 `uses_dual_buffer`）
2. `test_ch_step_dispatch_via_case_runner` — 同上 CH
3. `test_multiphase_prepare_diagnostics_baseline` — `reset_baseline=True` 後 mass_error = 0
4. `test_ch_prepare_diagnostics_baseline` — 同上 phi_drift = 0
5. `test_multiphase_diagnostics_check_convergence` — mass_error > tol → True
6. `test_ch_diagnostics_check_convergence` — phi_drift > tol → True
7. `test_rt_multiphase_benchmark_runner_builds` — `create_benchmark_runner("rayleigh_taylor_multiphase")` 回傳 CaseRunner
8. `test_rt_ch_benchmark_runner_builds` — 同上 CH
