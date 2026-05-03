# 🌊 CFD Taichi: Taichi 1.7.4 FVM / LBM Solver Toolset

[![Taichi](https://img.shields.io/badge/backend-Taichi-red.svg)](https://taichi-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**CFD Taichi** 是以 **Taichi 1.7.4** 實作的 **FVM / LBM solver toolkit**，用於二維 CFD 問題的研究、驗證、原型開發與可重現案例流程。專案主線已從單一 example script 轉向 **benchmark-first workflow**：

- benchmark registry
- `CaseRunner` 組裝與 diagnostics hook
- matrix regression / payload 輸出
- preset-based parameter sweep

目前版本：`0.2.0`

---

## 🚀 核心特性

- **雙 solver 架構**：同時提供 `lbm_taichi` 與 `fvm_taichi`，支援跨方法對照與擴充。
- **LBM 工具鏈**：MRT / ELBM / EMRT、Smagorinsky LES、邊界條件、粒子煙線與診斷監控。
- **FVM 工具鏈**：Euler / Navier-Stokes solver、網格生成、Cp / 流場 / 極線後處理。
- **工具化導向**：solver、邊界條件、diagnostics、幾何與可視化模組可被案例重用。
- **benchmark-first workflow**：案例註冊、CLI 執行、matrix regression 與 `.npy` payload 已成正式 surface。
- **可驗證流程**：強調 `mass_error`、殘差、力係數、shock 指標、解析解誤差與物理場判讀，而不是只看是否收斂。

---

## 📦 專案結構

```
cfd_taichi/
├── src/
│   ├── lbm_taichi/                   # LBM solver 與工具
│   ├── fvm_taichi/                   # FVM solver 與工具
│   └── cfd_taichi/                   # package / CLI 入口
├── examples/                         # LBM / FVM 案例腳本
├── tests/                            # 測試、回歸與驗證
└── docs/                             # 技術文件
```

---

## ⚙️ 安裝

```bash
uv sync
```

建議環境：

- Python 3.10.12
- Taichi 1.7.4
- Apple Silicon + Metal backend

---

## ▶️ 執行方式

**benchmark registry CLI**
```bash
uv run cfd-taichi benchmark list --payload-file benchmark_index.npy
uv run cfd-taichi benchmark poiseuille --arch cpu --steps 10 --sample-interval 5 --set ni=64 --set nj=32 --payload-file poiseuille_summary.npy
uv run cfd-taichi benchmark couette --arch cpu --steps 10 --sample-interval 5 --set ni=64 --set nj=32 --payload-file couette_summary.npy
uv run cfd-taichi benchmark cd_nozzle --arch cpu --steps 10 --sample-interval 5 --set ni=40 --set nj=16 --payload-file cd_nozzle_summary.npy
uv run cfd-taichi benchmark naca_euler --arch cpu --steps 10 --sample-interval 5 --set ni=80 --set nj=24 --set ma=0.3 --set aoa=5.0 --payload-file naca_euler_summary.npy
uv run cfd-taichi benchmark naca_ns --arch cpu --steps 10 --sample-interval 5 --set ni=80 --set nj=24 --set ma=0.12 --set re=300.0 --set aoa=2.0 --payload-file naca_ns_summary.npy
```

**preset-based sweep**
```bash
uv run python examples/naca0012_ns_sweep.py --sweep aoa --values 0,2,4 --steps 200 --output output_naca_ns_sweep
uv run python examples/transonic_bump_sweep.py --sweep ma --values 0.62,0.66,0.70 --steps 200 --output output_bump_sweep
```

兩類 sweep 目前都會重用 benchmark matrix 路徑，並在輸出目錄生成 `matrix_summary.npy`。

## ✅ Current Toolkit Surface

**正式 benchmark**

- `lid_driven_cavity`
- `transonic_bump_euler` / `transonic_bump`
- `cd_nozzle_euler` / `cd_nozzle`
- `naca0012_euler` / `naca_euler`
- `naca0012_ns` / `naca_ns`
- `poiseuille_flow_ns` / `poiseuille`
- `couette_flow_ns` / `couette`

**正式 preset**

- `build_naca0012_ns_sweep_preset`
- `build_transonic_bump_sweep_preset`

## 🆕 0.2.0 Highlights

- benchmark registry 已成主要入口，取代舊的 script-first 啟動路徑
- `CaseRunner` 已覆蓋 LBM、compressible FVM、incompressible FVM 三條主線
- benchmark CLI 已支援固定 schema 的 `list` / `run` 輸出與 `.npy` payload
- `naca0012_ns_sweep.py` 與 `transonic_bump_sweep.py` 已收斂成 preset-based sweep
- `example_loader` 與舊入口 `run.py` 已移除

## 🧭 Roadmap

**下一批 benchmark 候選**

- `flow_over_cylinder.py`：補齊 LBM bluff-body benchmark 主線
- `sod_shock_tube.py`：補齊 compressible FV 最小 regression 基準
- `backward_facing_step.py`：補齊分離/再附著與出口穩定性 benchmark
- `taylor_green_vortex.py`：補齊週期邊界與耗散 regression benchmark

**尚未完成的完整度**

- `transonic_bump_euler` acceptance 仍偏弱，尚未把 shock plateau / shock position 正式納入 registry gate
- preset 目前是 Python API 與 example 薄包裝，尚未有對應的 `preset list/run` CLI
- thermal / multiphase 路線仍停留在候選案例，尚未進入正式 benchmark surface

## 🗂️ Example Migration Status

目前 `examples/` 不再被視為主要入口；主流程以 benchmark registry 為準。下列盤點用來收斂下一波移除與 toolkit 化。

**保留並 toolkit 化**

- `lid_driven_cavity.py`：已完成 toolkit 化，保留為 LBM 核心 benchmark。
- `transonic_bump_euler.py`：已完成 toolkit 化，保留為 compressible FVM 核心 benchmark。
- `cd_nozzle_euler.py`：已完成 toolkit 化，保留為 compressible FVM nozzle benchmark。
- `poiseuille_flow_ns.py`：已完成 toolkit 化，保留為 incompressible FVM benchmark。
- `couette_flow_ns.py`：已完成 toolkit 化，保留為 incompressible FVM moving-wall benchmark。
- `sod_shock_tube.py`：可壓縮 FV 最小驗證單元，應整理成 core regression，而不是只留在 example。
- `naca0012_euler.py`：已完成 toolkit 化，保留為 compressible FVM external-aero benchmark。
- `naca0012_ns.py`：已完成 toolkit 化，保留為 viscous compressible FVM external-aero benchmark。
- `naca0012_ns_sweep.py`：保留為 benchmark matrix preset 的後處理薄包裝，不再維持獨立逐點 workflow。
- `transonic_bump_sweep.py`：保留為 benchmark matrix preset 的後處理薄包裝，不再維持獨立逐點 workflow。
- `flow_over_cylinder.py`：LBM 經典 bluff-body 驗證，應作為下一批 toolkit benchmark 候選。
- `backward_facing_step.py`：分離/再附著與 outlet 穩定性驗證具代表性，應納入 toolkit 路線。
- `taylor_green_vortex.py`：週期邊界與耗散驗證價值高，適合作為 LBM regression benchmark。
- `kelvin_helmholtz.py`：非定常剪切層基準，有研究價值，應保留。
- `rayleigh_benard.py`：對 thermal LBM 有獨立驗證價值，若保留熱流路線，應 toolkit 化。
- `rayleigh_taylor_ch.py`：若 CH multiphase solver 要維持為正式能力，這是較合理的保留案例。
- `rayleigh_taylor_ch_sweep.py`：保留前提是 `rayleigh_taylor_ch.py` 正式化；否則應跟著刪除。
- `rayleigh_taylor_multiphase.py`：若 Shan-Chen multiphase 仍在 roadmap 內，保留並等待 toolkit 化。

**已刪除**

- `airfoil.py`：已移除。多段翼 LBM 高升力案例過度特化，且與目前 benchmark-first 主線脫節。
- `rayleigh_taylor.py`：已移除。單相 Boussinesq 近似版本與 CH / multiphase 路線重疊，且物理定位較弱。

---

## 📈 可視化

```bash
uv run python src/lbm_taichi/utils/visualization.py output_lid_driven_cavity --gif --type velocity
```

---

## 🧩 套件匯入

```python
from lbm_taichi import LBMSolver, BoundaryConditions, Diagnostics
from fvm_taichi import (
    CompressibleEulerSolver,
    CompressibleNavierStokesSolver,
)
from cfd_taichi import (
    BenchmarkMatrixEntry,
    BoundaryConditionDescriptor,
    CartesianGrid2D,
    CaseRunner,
    build_naca0012_ns_sweep_preset,
    build_transonic_bump_sweep_preset,
    build_benchmark_matrix_payload,
    create_benchmark_runner,
    create_solver,
    run_benchmark_matrix,
    save_benchmark_matrix_payload,
)

solver = create_solver(
    method="fvm",
    equation="navier_stokes",
    regime="compressible",
    ni=64,
    nj=32,
    re=100.0,
    u_ref=0.1,
    length_scale=1.0,
)

runner = CaseRunner(
    name="inc-channel",
    method="fvm",
    equation="navier_stokes",
    regime="incompressible",
    grid=CartesianGrid2D(ni=64, nj=32, dx=1.0 / 64, dy=1.0 / 32),
    solver_kwargs={"re": 100.0, "u_ref": 0.1, "length_scale": 1.0},
    boundary_conditions=[
        BoundaryConditionDescriptor.periodic("x"),
        BoundaryConditionDescriptor.no_slip("bottom"),
        BoundaryConditionDescriptor.no_slip("top", u_wall=0.1),
    ],
)
runner.build_solver()
runner.configure()
runner.save_state_file("state_000000.npy")
runner.load_state("state_000000.npy")
runner.save_history_file("history.npy", params={"case": "demo"})
runner.load_history("history.npy")

ldc = create_benchmark_runner("lid_driven_cavity", res=64, re=100.0)
bump = create_benchmark_runner("transonic_bump_euler", ni=80, nj=24, ma=0.70)
nozzle = create_benchmark_runner("cd_nozzle", ni=48, nj=16, ma_init=0.12)
naca = create_benchmark_runner("naca_euler", ni=80, nj=24, ma=0.3, aoa=5.0)
naca_ns = create_benchmark_runner("naca_ns", ni=80, nj=24, ma=0.12, re=300.0, aoa=2.0)
poiseuille = create_benchmark_runner("poiseuille", ni=64, nj=32, re=20.0, u_max=0.05)
couette = create_benchmark_runner("couette", ni=64, nj=32, re=100.0, u_top=0.1)

matrix = run_benchmark_matrix(
    [
        BenchmarkMatrixEntry(
            benchmark="lid_driven_cavity",
            steps=10,
            sample_interval=5,
            overrides={"res": 64, "re": 100.0},
        ),
        BenchmarkMatrixEntry(
            benchmark="transonic_bump_euler",
            steps=5,
            sample_interval=5,
            overrides={"ni": 80, "nj": 24, "ma": 0.70},
        ),
        BenchmarkMatrixEntry(
            benchmark="cd_nozzle_euler",
            steps=5,
            sample_interval=5,
            overrides={"ni": 48, "nj": 16, "ma_init": 0.12},
        ),
        BenchmarkMatrixEntry(
            benchmark="naca0012_euler",
            steps=5,
            sample_interval=5,
            overrides={"ni": 80, "nj": 24, "ma": 0.3, "aoa": 5.0},
        ),
        BenchmarkMatrixEntry(
            benchmark="naca0012_ns",
            steps=5,
            sample_interval=5,
            overrides={"ni": 80, "nj": 24, "ma": 0.12, "re": 300.0, "aoa": 2.0},
        ),
        BenchmarkMatrixEntry(
            benchmark="poiseuille",
            steps=5,
            sample_interval=5,
            overrides={"ni": 64, "nj": 32, "re": 20.0, "u_max": 0.05},
        ),
        BenchmarkMatrixEntry(
            benchmark="couette",
            steps=5,
            sample_interval=5,
            overrides={"ni": 64, "nj": 32, "re": 100.0, "u_top": 0.1},
        ),
    ]
)
matrix_payload = build_benchmark_matrix_payload(matrix)
save_benchmark_matrix_payload("benchmark_matrix.npy", matrix)
# matrix_payload["acceptance_passed"] / ["acceptance_failed_count"]
# 可直接作為 regression gate

naca_sweep = build_naca0012_ns_sweep_preset(
    sweep="aoa",
    values=[0.0, 2.0, 4.0],
    ni=80,
    nj=24,
    ma=0.12,
    re=300.0,
)
bump_sweep = build_transonic_bump_sweep_preset(
    sweep="ma",
    values=[0.62, 0.66, 0.70],
    ni=80,
    nj=24,
)
```

---

## 🧪 經典案例

**LBM**

- **Lid-Driven Cavity**：邊界與收斂驗證
- **Flow Over Cylinder**：卡門渦街與 Strouhal
- **Kelvin-Helmholtz**：剪切不穩定性

**FVM**

- **CD Nozzle Euler**：收縮擴張噴嘴與背壓問題
- **NACA0012 Euler / Navier-Stokes**：翼型壓力分佈與黏性效應
- **Transonic Bump**：可壓縮流與 shock-like 壓縮區驗證
- **Poiseuille Flow NS**：不可壓縮 projection solver 與解析拋物線剖面驗證
- **Couette Flow NS**：不可壓縮 moving-wall 通道與解析線性剖面驗證

---

## 📘 文件

- `docs/BOUNDARY_CONDITIONS.md`：邊界條件使用方式與研究檢核協議
- `AGENTS.md`：專案定位、研究夥伴模式與開發規範
- `tests/README.md`：測試與驗證腳本說明

---

## 🙏 致謝

特別感謝 [Taichi 團隊](https://github.com/taichi-dev) 提供高效能計算框架。

**開發者**: latteine1217  \
**版本**: 0.2.0  \
**維護狀態**: ✅ 主動開發中  \
**授權**: MIT License
