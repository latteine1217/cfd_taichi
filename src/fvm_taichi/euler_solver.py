"""
EulerSolver — 2D 可壓縮 Euler FV 求解器
==========================================

What: 結構化網格上的有限體積 Euler 方程求解器
Why:  LBM (D2Q9) 在 Ma > 0.3 時精度不足；FVM + Riemann solver 是
      跨音速/超音速翼型的工業標準路線
When: 適用 Ma 0.3–5.0 的無黏可壓縮流（翼型、激波管、跨音速通道）

數值方法：
  空間重建：MUSCL（minmod limiter，原始量空間）
  Riemann solver：HLLC（Toro 1994）
  時間推進：SSP-RK3（Shu–Osher）
  正性保護：ρ/p ≤ 0 → 降回一階

索引規格（NG=2，兩層 ghost）：
  總格點  : (ni+4) × (nj+4)
  Interior: [NG, NG+ni) × [NG, NG+nj)
  Face-i  : [NG-1, NG+ni) × [NG, NG+nj)   （共 ni+1 條 i-faces）
  Face-j  : [NG, NG+ni) × [NG-1, NG+nj)   （共 nj+1 條 j-faces）
"""

import taichi as ti
import numpy as np


@ti.data_oriented
class EulerSolver:
    """
    2D 可壓縮 Euler FV 求解器（結構化網格）

    使用方式（直角網格）：
        solver = EulerSolver(ni=200, nj=100)
        solver.set_cartesian_grid(dx=0.005, dy=0.005)
        solver.init_uniform(rho=1.0, u=0.3, v=0.0, p=0.714)
        solver.step()
        rho, u, v, p = solver.get_primitive()
    """

    solver_family = "fvm"
    equation_set = "euler"
    regime = "compressible"

    NG = 2       # ghost 層數（MUSCL 二階 stencil 需要 2 層）
    GAMMA = 1.4  # 比熱比（空氣，預設）

    def __init__(self, ni: int, nj: int, gamma: float = 1.4, cfl: float = 0.45):
        """
        Args:
            ni:    i 方向 interior cell 數
            nj:    j 方向 interior cell 數
            gamma: 比熱比（空氣 1.4，雙原子氣體）
            cfl:   CFL 數（顯式穩定，建議 0.4–0.5）
        """
        self.ni    = ni
        self.nj    = nj
        self.gamma = gamma
        self.cfl   = cfl

        NG = self.NG
        NI = ni + 2 * NG   # i 方向總格點數（含 ghost）
        NJ = nj + 2 * NG   # j 方向總格點數（含 ghost）
        self.NI = NI
        self.NJ = NJ

        # ── 守恆量 U = [ρ, ρu, ρv, E] ──────────────────────────
        self.U  = ti.Vector.field(4, ti.f32, shape=(NI, NJ))
        self.U0 = ti.Vector.field(4, ti.f32, shape=(NI, NJ))  # RK3 步起點
        self.U1 = ti.Vector.field(4, ti.f32, shape=(NI, NJ))  # RK3 中間步 1
        self.U2 = ti.Vector.field(4, ti.f32, shape=(NI, NJ))  # RK3 中間步 2

        # ── 原始量 W = [ρ, u, v, p]（limiter 在原始量空間操作）──
        self.W = ti.Vector.field(4, ti.f32, shape=(NI, NJ))

        # ── Face 重建後的左/右狀態 ────────────────────────────────
        # i-face(i+½, j)：between cell(i,j) and cell(i+1,j)
        self.WL_i = ti.Vector.field(4, ti.f32, shape=(NI, NJ))
        self.WR_i = ti.Vector.field(4, ti.f32, shape=(NI, NJ))
        # j-face(i, j+½)：between cell(i,j) and cell(i,j+1)
        self.WL_j = ti.Vector.field(4, ti.f32, shape=(NI, NJ))
        self.WR_j = ti.Vector.field(4, ti.f32, shape=(NI, NJ))

        # ── Face 通量 ─────────────────────────────────────────────
        self.Flux_i = ti.Vector.field(4, ti.f32, shape=(NI, NJ))
        self.Flux_j = ti.Vector.field(4, ti.f32, shape=(NI, NJ))

        # ── 殘差 R：dU/dt = −R ────────────────────────────────────
        self.R = ti.Vector.field(4, ti.f32, shape=(NI, NJ))
        self.R_tmp = ti.Vector.field(4, ti.f32, shape=(NI, NJ))

        # ── 網格幾何（precomputed，set_*_grid() 填入）─────────────
        # Face 向量 S = n̂·ΔS（有方向、含面積資訊）
        # S_i[i,j] = i-face(i+½, j) 的面向量（指向 +i 方向）
        self.S_i = ti.Vector.field(2, ti.f32, shape=(NI, NJ))
        self.S_j = ti.Vector.field(2, ti.f32, shape=(NI, NJ))
        # 控制體積面積
        self.vol = ti.field(ti.f32, shape=(NI, NJ))

        # ── 時間步（全域 reduce）─────────────────────────────────
        self.dt_field = ti.field(ti.f32, shape=())
        self.dt_min_field = ti.field(ti.f32, shape=())
        self.dt_max_field = ti.field(ti.f32, shape=())
        self.dt_sum_field = ti.field(ti.f32, shape=())
        self.dt_local = ti.field(ti.f32, shape=(NI, NJ))
        self.bad_state_flat = ti.field(ti.i32, shape=())
        self.time_marching_mode_field = ti.field(ti.i32, shape=())
        self.pseudo_iter_field = ti.field(ti.i32, shape=())
        self.pseudo_cfl_start_field = ti.field(ti.f32, shape=())
        self.pseudo_cfl_ramp_steps_field = ti.field(ti.i32, shape=())
        self.pseudo_cfl_eff_field = ti.field(ti.f32, shape=())
        self.pseudo_cfl_target_field = ti.field(ti.f32, shape=())
        self.pseudo_precond_ref_mach_field = ti.field(ti.f32, shape=())
        self.pseudo_precond_min_scale_field = ti.field(ti.f32, shape=())
        self.pseudo_precond_scale_min_field = ti.field(ti.f32, shape=())
        self.pseudo_precond_scale_max_field = ti.field(ti.f32, shape=())
        self.pseudo_precond_scale_sum_field = ti.field(ti.f32, shape=())
        self.residual_smoothing_eps_field = ti.field(ti.f32, shape=())
        self.residual_smoothing_base_eps_field = ti.field(ti.f32, shape=())
        self.residual_smoothing_max_eps_field = ti.field(ti.f32, shape=())
        self.residual_smoothing_passes_field = ti.field(ti.i32, shape=())
        self.pseudo_adaptive_enabled_field = ti.field(ti.i32, shape=())
        self.pseudo_adapt_interval_field = ti.field(ti.i32, shape=())
        self.pseudo_adapt_growth_field = ti.field(ti.f32, shape=())
        self.pseudo_adapt_shrink_field = ti.field(ti.f32, shape=())
        self.pseudo_adapt_target_ratio_field = ti.field(ti.f32, shape=())
        self.pseudo_adapt_fail_ratio_field = ti.field(ti.f32, shape=())
        self.pseudo_cfl_min_field = ti.field(ti.f32, shape=())
        self.pseudo_cfl_max_field = ti.field(ti.f32, shape=())
        self.pseudo_latest_residual_field = ti.field(ti.f32, shape=())
        self.pseudo_reference_residual_field = ti.field(ti.f32, shape=())
        self.pseudo_residual_ratio_field = ti.field(ti.f32, shape=())
        self.first_order_recon_field = ti.field(ti.i32, shape=())
        self._time_marching_mode = "global"
        self.time_marching_mode_field[None] = 0
        self.pseudo_iter_field[None] = 0
        self.pseudo_cfl_start_field[None] = float(cfl)
        self.pseudo_cfl_ramp_steps_field[None] = 0
        self.pseudo_cfl_eff_field[None] = float(cfl)
        self.pseudo_cfl_target_field[None] = float(cfl)
        self.pseudo_precond_ref_mach_field[None] = 1.0
        self.pseudo_precond_min_scale_field[None] = 1.0
        self.pseudo_precond_scale_min_field[None] = 1.0
        self.pseudo_precond_scale_max_field[None] = 1.0
        self.pseudo_precond_scale_sum_field[None] = 0.0
        self.residual_smoothing_eps_field[None] = 0.0
        self.residual_smoothing_base_eps_field[None] = 0.0
        self.residual_smoothing_max_eps_field[None] = 0.0
        self.residual_smoothing_passes_field[None] = 0
        self.pseudo_adaptive_enabled_field[None] = 0
        self.pseudo_adapt_interval_field[None] = 20
        self.pseudo_adapt_growth_field[None] = 1.03
        self.pseudo_adapt_shrink_field[None] = 0.7
        self.pseudo_adapt_target_ratio_field[None] = 0.995
        self.pseudo_adapt_fail_ratio_field[None] = 1.005
        self.pseudo_cfl_min_field[None] = float(cfl)
        self.pseudo_cfl_max_field[None] = float(cfl)
        self.pseudo_latest_residual_field[None] = -1.0
        self.pseudo_reference_residual_field[None] = -1.0
        self.pseudo_residual_ratio_field[None] = 1.0
        self.first_order_recon_field[None] = 0

        # 幾何是否已設定（guard）
        self._grid_ready = False

        # 邊界條件旗標（預設 Neumann outflow）
        self._periodic_i     = False
        self._periodic_j     = False
        self._wall_j_min     = False   # j_min 滑動壁（翼型表面）
        self._wall_j_max     = False   # j_max 滑動壁（通道上壁）
        self._far_field_j_max = False  # j_max 特徵遠場 BC
        self._nozzle_inlet_i_min = False   # i_min nozzle 入口特徵 BC（p0/T0）
        self._nozzle_outlet_i_max = False  # i_max nozzle 出口特徵 BC（pb）

        # 遠場自由流狀態（供特徵遠場 BC 使用）
        self._rho_inf = 1.0
        self._u_inf   = 0.0
        self._v_inf   = 0.0
        self._p_inf   = 1.0
        self._nozzle_p0 = 1.0
        self._nozzle_t0 = 1.0
        self._nozzle_mach_in = 0.2
        self._nozzle_back_pressure = 1.0
        self._nozzle_inlet_relax = 0.35
        self._nozzle_outlet_relax = 0.35

    # ── 公開介面：網格設定 ────────────────────────────────────────

    def set_cartesian_grid(self, dx: float, dy: float):
        """
        設定均勻直角網格幾何

        What: 所有 i-face 面積 = dy，所有 j-face 面積 = dx，
              所有 cell 體積 = dx*dy，法向量為軸向
        Why:  直角網格是曲線網格的特例，用相同的幾何 field 表示，
              後續換成 C-grid 只需替換此方法
        """
        self._fill_cartesian_geometry(dx, dy)
        self._grid_ready = True

    def set_periodic_bc(self, i_dir: bool = False, j_dir: bool = False):
        """
        設定周期性邊界條件

        What: 在 i 或 j 方向啟用周期性 BC，取代預設 Neumann outflow
        Why:  isentropic vortex、Taylor-Green 等週期性解析測試需要
        When: set_cartesian_grid() 之後、模擬開始前呼叫

        Args:
            i_dir: True → i 方向周期（x 方向）
            j_dir: True → j 方向周期（y 方向）
        """
        self._periodic_i = i_dir
        self._periodic_j = j_dir

    def set_slip_wall_j_min(self):
        """
        將 j_min 邊界設為無黏滑動壁（Euler 翼型表面）

        What: ghost cells 以鏡像法線速度填充，切向速度保留，壓力/密度外插
        Why:  無黏壁面條件 (v_n=0)；O-grid 中 j=NG 緊鄰翼型表面
        """
        self._wall_j_min = True

    def set_slip_wall_j_max(self):
        """
        將 j_max 邊界設為無黏滑動壁（Euler 通道上壁）

        What: ghost cells 以鏡像法線速度填充，切向速度保留，壓力/密度外插
        Why:  transonic bump / 通道案例需要上下壁皆為不可穿透的 inviscid wall
        """
        self._wall_j_max = True

    def set_far_field_j_max(self,
                            rho: float, u: float, v: float, p: float):
        """
        將 j_max 邊界設為特徵遠場 BC（Riemann invariant，亞音速）

        What: 以 Riemann 不變量區分入射/出射特徵，
              出射取 interior 值，入射取自由流值
        Why:  純 Neumann 在非對稱來流（AOA ≠ 0）或長時模擬中會引入虛假反射；
              特徵 BC 允許擾動自然離開計算域
        When: 適用亞音速遠場（|u_n| < c），O-grid j_max 遠場邊界

        Args:
            rho, u, v, p: 自由流原始量
        """
        self._far_field_j_max = True
        self._rho_inf = float(rho)
        self._u_inf   = float(u)
        self._v_inf   = float(v)
        self._p_inf   = float(p)

    def set_nozzle_inlet_i_min(
            self, p0: float, t0: float, mach_in: float = 0.2, relax: float = 0.35):
        """
        設定 i_min 為 nozzle 入口特徵 BC（總壓/總溫控制）

        What: 入口使用特徵關係，亞音速由 p0/T0 + R- 重建；
              `mach_in` 僅在超音速入口時作為 fallback 目標值
        Why:  內流 nozzle 的控制量是 reservoir 總壓/總溫，而非固定靜態原始量
        When: converging-diverging nozzle 內流案例
        """
        if p0 <= 0.0 or t0 <= 0.0:
            raise ValueError(f"p0 and t0 must be positive, got p0={p0}, t0={t0}")
        if not (0.0 < relax <= 1.0):
            raise ValueError(f"relax must satisfy 0 < relax <= 1, got {relax}")
        self._nozzle_inlet_i_min = True
        self._periodic_i = False
        self._nozzle_p0 = float(p0)
        self._nozzle_t0 = float(t0)
        self._nozzle_mach_in = float(mach_in if mach_in > 0.0 else 0.2)
        self._nozzle_inlet_relax = float(relax)

    def set_nozzle_outlet_i_max(self, back_pressure: float, relax: float = 0.35):
        """
        設定 i_max 為 nozzle 出口特徵 BC（背壓控制）

        What: 亞音速出口以指定背壓 pb 注入入射特徵；超音速出口退化成外插
        Why:  pb/p0 決定 choking 與超音速段發展，是 nozzle 核心控制參數
        When: converging-diverging nozzle 內流案例
        """
        if back_pressure <= 0.0:
            raise ValueError(f"back_pressure must be positive, got {back_pressure}")
        if not (0.0 < relax <= 1.0):
            raise ValueError(f"relax must satisfy 0 < relax <= 1, got {relax}")
        self._nozzle_outlet_i_max = True
        self._periodic_i = False
        self._nozzle_back_pressure = float(back_pressure)
        self._nozzle_outlet_relax = float(relax)

    def set_curvilinear_grid(self, x_node: np.ndarray, y_node: np.ndarray):
        """
        設定曲線結構網格幾何（含 ghost cells）

        What: 從節點座標陣列計算面向量 S_i, S_j 與控制體積 vol
        Why:  翼型 O/C-grid 需要非直角網格；面向量公式與 Cartesian 相同，
              只是數值不再是常數

        Args:
            x_node, y_node: shape (NI+1, NJ+1) 節點 x/y 座標
                            NI = ni + 2*NG, NJ = nj + 2*NG
                            節點 (i, j) 為 cell (i, j) 的左下角

        Geometry（面向量正向約定）：
          S_i[i,j] = i-face(i+½,j)，從 node(i+1,j) → node(i+1,j+1)，指向 +i
          S_j[i,j] = j-face(i,j+½)，從 node(i,j+1) → node(i+1,j+1)，指向 +j
          vol[i,j] = quad 面積（對角線叉積/2）
        """
        assert x_node.shape == (self.NI + 1, self.NJ + 1), \
            f"Expected ({self.NI+1}, {self.NJ+1}), got {x_node.shape}"
        xn = np.ascontiguousarray(x_node, dtype=np.float32)
        yn = np.ascontiguousarray(y_node, dtype=np.float32)
        self._fill_curvilinear_geometry(xn, yn)
        self._grid_ready = True

    # ── 公開介面：初始化 ──────────────────────────────────────────

    def init_uniform(self, rho: float, u: float, v: float, p: float):
        """
        以均勻原始量初始化整個計算域（含 ghost cells）

        Args:
            rho: 密度
            u:   x 速度
            v:   y 速度
            p:   壓力
        """
        self._init_uniform_kernel(float(rho), float(u), float(v), float(p))

    # ── 公開介面：時間推進 ────────────────────────────────────────

    def step(self):
        """
        執行一個 SSP-RK3 時間步

        Why SSP-RK3：Strong Stability Preserving，保持 TVD 性質，
                     在激波附近不引入新的極值
        """
        assert self._grid_ready, "請先呼叫 set_*_grid() 設定網格"

        self._prepare_local_pseudo_controls_for_step()
        self._compute_dt()
        ti.sync()
        dt = float(self.dt_field[None])

        self._save_U0()

        # Stage 1：R(U^n) → U1
        self._eval_residual()
        if self._time_marching_mode == "local_pseudo":
            self._rk_stage1_local()
        else:
            self._rk_stage1(dt)
        self._assert_positive_state(self.U1, "RK stage 1")
        self._set_state(self.U1)

        # Stage 2：R(U1) → U2
        self._eval_residual()
        if self._time_marching_mode == "local_pseudo":
            self._rk_stage2_local()
        else:
            self._rk_stage2(dt)
        self._assert_positive_state(self.U2, "RK stage 2")
        self._set_state(self.U2)

        # Stage 3：R(U2) → U^{n+1}
        self._eval_residual()
        if self._time_marching_mode == "local_pseudo":
            self._rk_stage3_local()
        else:
            self._rk_stage3(dt)
        self._assert_positive_state(self.U, "RK stage 3")
        self._update_primitive()
        self._update_ghost()
        if self._time_marching_mode == "local_pseudo":
            residual = float(self._compute_residual_norm())
            self._record_local_pseudo_residual(residual)
            self.pseudo_iter_field[None] += 1

        return dt

    # ── 公開介面：資料讀取 ────────────────────────────────────────

    def get_primitive(self) -> tuple:
        """
        回傳 interior cells 的原始量（numpy）

        Returns:
            (rho, u, v, p) 各為 shape (ni, nj) 的 float32 ndarray
        """
        NG = self.NG
        W_np = self.W.to_numpy()
        interior = W_np[NG:NG+self.ni, NG:NG+self.nj]
        return (interior[:,:,0], interior[:,:,1],
                interior[:,:,2], interior[:,:,3])

    def get_fields(self) -> dict[str, np.ndarray]:
        """
        回傳標準化場資料。
        """
        rho, u, v, p = self.get_primitive()
        vel = np.stack([u, v], axis=2)
        speed = np.sqrt(u * u + v * v)
        c = np.sqrt(np.maximum(self.gamma * p / np.maximum(rho, 1e-12), 1e-12))
        mach = speed / np.maximum(c, 1e-12)
        return {
            "rho": rho,
            "u": vel,
            "p": p,
            "mach": mach,
        }

    def get_diagnostics(self) -> dict:
        """
        回傳結構化診斷量。
        """
        fields = self.get_fields()
        dt_diag = self.get_dt_diagnostics()
        return {
            **dt_diag,
            "rho_min": float(np.min(fields["rho"])),
            "rho_max": float(np.max(fields["rho"])),
            "p_min": float(np.min(fields["p"])),
            "p_max": float(np.max(fields["p"])),
            "u_max": float(np.max(np.linalg.norm(fields["u"], axis=2))),
            "mach_max": float(np.max(fields["mach"])),
        }

    def get_dt(self) -> float:
        """回傳最後一步的時間步長"""
        return float(self.dt_field[None])

    def set_time_marching_mode(self, mode: str = "global"):
        """
        設定時間推進模式

        What: 支援 `global` 與 `local_pseudo` 兩種時間步策略
        Why:  steady-state acceleration 需要 local pseudo-time stepping，
              但 physical-time case 仍必須保留全域單一 dt
        """
        if mode not in {"global", "local_pseudo"}:
            raise ValueError(
                "time marching mode must be 'global' or 'local_pseudo', "
                f"got {mode}"
            )
        self._time_marching_mode = mode
        self.time_marching_mode_field[None] = 0 if mode == "global" else 1
        self.pseudo_iter_field[None] = 0
        self.pseudo_cfl_eff_field[None] = float(self.cfl)
        self.pseudo_cfl_target_field[None] = float(self.cfl)
        self.pseudo_precond_scale_min_field[None] = 1.0
        self.pseudo_precond_scale_max_field[None] = 1.0
        self.pseudo_precond_scale_sum_field[None] = 0.0
        self.pseudo_latest_residual_field[None] = -1.0
        self.pseudo_reference_residual_field[None] = -1.0
        self.pseudo_residual_ratio_field[None] = 1.0

    def set_pseudo_time_controls(
            self,
            cfl_start: float | None = None,
            ramp_steps: int = 0,
            precond_ref_mach: float = 1.0,
            precond_min_scale: float = 1.0):
        """
        設定 local pseudo-time 的 CFL ramp 與 acoustic scaling

        What: 只影響 `local_pseudo` 模式下的 local dt 估算
        Why:  steady acceleration 需要先保守啟動，再逐步升 CFL；
              低馬赫 pseudo-time 則可縮減聲速項造成的 stiffness
        """
        start = self.cfl if cfl_start is None else float(cfl_start)
        if start <= 0.0 or start > self.cfl:
            raise ValueError(
                f"pseudo-time cfl_start must satisfy 0 < cfl_start <= base CFL ({self.cfl}), "
                f"got {start}"
            )
        if ramp_steps < 0:
            raise ValueError(f"ramp_steps must be >= 0, got {ramp_steps}")
        if precond_ref_mach <= 0.0:
            raise ValueError(f"precond_ref_mach must be positive, got {precond_ref_mach}")
        if not (0.0 < precond_min_scale <= 1.0):
            raise ValueError(
                "precond_min_scale must satisfy 0 < precond_min_scale <= 1, "
                f"got {precond_min_scale}"
            )
        self.pseudo_cfl_start_field[None] = start
        self.pseudo_cfl_ramp_steps_field[None] = int(ramp_steps)
        self.pseudo_precond_ref_mach_field[None] = float(precond_ref_mach)
        self.pseudo_precond_min_scale_field[None] = float(precond_min_scale)
        self.pseudo_cfl_target_field[None] = start
        self.pseudo_iter_field[None] = 0
        self.pseudo_latest_residual_field[None] = -1.0
        self.pseudo_reference_residual_field[None] = -1.0
        self.pseudo_residual_ratio_field[None] = 1.0

    def set_residual_smoothing(self, epsilon: float = 0.0, passes: int = 0):
        """
        設定 local pseudo-time residual smoothing

        What: 在每個 pseudo-time RK stage 對空間殘差做 Jacobi-like 平滑
        Why:  顯式 steady marching 常被高頻殘差模態卡住；適度 smoothing
              可降低 odd-even / grid-scale stiffness
        """
        if epsilon < 0.0:
            raise ValueError(f"residual smoothing epsilon must be >= 0, got {epsilon}")
        if passes < 0:
            raise ValueError(f"residual smoothing passes must be >= 0, got {passes}")
        if passes == 0:
            epsilon = 0.0
        self.residual_smoothing_eps_field[None] = float(epsilon)
        self.residual_smoothing_base_eps_field[None] = float(epsilon)
        self.residual_smoothing_max_eps_field[None] = float(epsilon)
        self.residual_smoothing_passes_field[None] = int(passes)

    def set_adaptive_pseudo_strategy(
            self,
            enabled: bool = False,
            cfl_min: float | None = None,
            cfl_max: float | None = None,
            growth: float = 1.03,
            shrink: float = 0.7,
            target_ratio: float = 0.995,
            fail_ratio: float = 1.005,
            interval: int = 20,
            smoothing_max_eps: float | None = None):
        """
        設定 local pseudo-time 的自適應 CFL / smoothing 策略

        What: 每隔固定 pseudo iterations 依 residual ratio 調整 CFL 與 smoothing
        Why:  固定常數對不同格點區域與收斂階段通常不是最佳選擇；
              自適應策略可在穩定與加速之間自動折衷
        """
        if interval <= 0:
            raise ValueError(f"adaptive pseudo interval must be positive, got {interval}")
        if growth < 1.0:
            raise ValueError(f"adaptive pseudo growth must be >= 1, got {growth}")
        if not (0.0 < shrink <= 1.0):
            raise ValueError(f"adaptive pseudo shrink must satisfy 0 < shrink <= 1, got {shrink}")
        if not (0.0 < target_ratio <= fail_ratio):
            raise ValueError(
                "adaptive pseudo ratios must satisfy 0 < target_ratio <= fail_ratio, "
                f"got {target_ratio}, {fail_ratio}"
            )

        cfl_min_eff = float(self.pseudo_cfl_start_field[None]) if cfl_min is None else float(cfl_min)
        cfl_max_eff = float(self.cfl) if cfl_max is None else float(cfl_max)
        if not (0.0 < cfl_min_eff <= cfl_max_eff):
            raise ValueError(
                "adaptive pseudo CFL bounds must satisfy 0 < cfl_min <= cfl_max, "
                f"got {cfl_min_eff}, {cfl_max_eff}"
            )
        base_eps = float(self.residual_smoothing_base_eps_field[None])
        smoothing_max_eff = base_eps if smoothing_max_eps is None else float(smoothing_max_eps)
        if smoothing_max_eff < base_eps:
            raise ValueError(
                "adaptive pseudo smoothing_max_eps must be >= base residual smoothing epsilon, "
                f"got {smoothing_max_eff} < {base_eps}"
            )

        self.pseudo_adaptive_enabled_field[None] = 1 if enabled else 0
        self.pseudo_adapt_interval_field[None] = int(interval)
        self.pseudo_adapt_growth_field[None] = float(growth)
        self.pseudo_adapt_shrink_field[None] = float(shrink)
        self.pseudo_adapt_target_ratio_field[None] = float(target_ratio)
        self.pseudo_adapt_fail_ratio_field[None] = float(fail_ratio)
        self.pseudo_cfl_min_field[None] = cfl_min_eff
        self.pseudo_cfl_max_field[None] = cfl_max_eff
        self.pseudo_cfl_target_field[None] = cfl_min_eff
        self.residual_smoothing_max_eps_field[None] = smoothing_max_eff
        self.residual_smoothing_eps_field[None] = base_eps
        self.pseudo_latest_residual_field[None] = -1.0
        self.pseudo_reference_residual_field[None] = -1.0
        self.pseudo_residual_ratio_field[None] = 1.0
        self.pseudo_iter_field[None] = 0

    def get_time_marching_mode(self) -> str:
        """回傳目前時間推進模式"""
        return self._time_marching_mode

    def set_spatial_reconstruction(self, order: str = "second"):
        """
        設定空間重建階數

        What: 支援 `second`（MUSCL）與 `first`（piecewise-constant）
        Why:  部分 steady nozzle/transonic case 會出現低頻震盪；
              baseline 建立時可先用一階耗散穩定收斂
        """
        if order not in {"first", "second"}:
            raise ValueError(f"spatial reconstruction order must be 'first' or 'second', got {order}")
        self.first_order_recon_field[None] = 1 if order == "first" else 0

    def get_dt_diagnostics(self) -> dict:
        """
        回傳 local/global dt 統計量

        What: 提供最小/平均/最大 local dt 與目前步進模式
        Why:  pseudo-time case 不能只看單一 global dt；需要知道局部 dt 分佈
        """
        count = max(self.ni * self.nj, 1)
        if self._time_marching_mode == "local_pseudo":
            scale_min = float(self.pseudo_precond_scale_min_field[None])
            scale_mean = float(self.pseudo_precond_scale_sum_field[None]) / count
            scale_max = float(self.pseudo_precond_scale_max_field[None])
        else:
            scale_min = 1.0
            scale_mean = 1.0
            scale_max = 1.0
        return {
            "mode": self._time_marching_mode,
            "dt_min": float(self.dt_min_field[None]),
            "dt_mean": float(self.dt_sum_field[None]) / count,
            "dt_max": float(self.dt_max_field[None]),
            "dt_global": float(self.dt_field[None]),
            "pseudo_iter": int(self.pseudo_iter_field[None]),
            "pseudo_cfl": float(self.pseudo_cfl_eff_field[None]),
            "pseudo_cfl_target": float(self.pseudo_cfl_target_field[None]),
            "pseudo_precond_scale_min": scale_min,
            "pseudo_precond_scale_mean": scale_mean,
            "pseudo_precond_scale_max": scale_max,
            "residual_smoothing_eps": float(self.residual_smoothing_eps_field[None]),
            "residual_smoothing_base_eps": float(self.residual_smoothing_base_eps_field[None]),
            "residual_smoothing_max_eps": float(self.residual_smoothing_max_eps_field[None]),
            "residual_smoothing_passes": int(self.residual_smoothing_passes_field[None]),
            "pseudo_adaptive_enabled": bool(self.pseudo_adaptive_enabled_field[None]),
            "pseudo_residual_ratio": float(self.pseudo_residual_ratio_field[None]),
        }

    def _prepare_local_pseudo_controls_for_step(self):
        """
        在每個 local pseudo-time step 前更新自適應控制

        Why:  讓 `get_dt_diagnostics()` 回報的是本步真正使用的 CFL / smoothing，
              而不是下一步預計使用的值
        """
        if self._time_marching_mode != "local_pseudo":
            return
        if self.pseudo_adaptive_enabled_field[None] == 1:
            interval = int(self.pseudo_adapt_interval_field[None])
            next_iter = int(self.pseudo_iter_field[None]) + 1
            latest = float(self.pseudo_latest_residual_field[None])
            reference = float(self.pseudo_reference_residual_field[None])
            if latest > 0.0 and reference <= 0.0:
                self.pseudo_reference_residual_field[None] = latest
                self.pseudo_residual_ratio_field[None] = 1.0
            elif latest > 0.0 and reference > 0.0 and next_iter > 1 and ((next_iter - 1) % interval == 0):
                ratio = latest / max(reference, 1e-30)
                self.pseudo_residual_ratio_field[None] = ratio

                cfl_target = float(self.pseudo_cfl_target_field[None])
                cfl_min = float(self.pseudo_cfl_min_field[None])
                cfl_max = float(self.pseudo_cfl_max_field[None])
                growth = float(self.pseudo_adapt_growth_field[None])
                shrink = float(self.pseudo_adapt_shrink_field[None])
                target_ratio = float(self.pseudo_adapt_target_ratio_field[None])
                fail_ratio = float(self.pseudo_adapt_fail_ratio_field[None])

                eps = float(self.residual_smoothing_eps_field[None])
                base_eps = float(self.residual_smoothing_base_eps_field[None])
                max_eps = float(self.residual_smoothing_max_eps_field[None])

                if ratio <= target_ratio:
                    cfl_target = min(cfl_target * growth, cfl_max)
                    if eps > base_eps:
                        eps = max(base_eps, 0.9 * eps)
                elif ratio >= fail_ratio:
                    cfl_target = max(cfl_target * shrink, cfl_min)
                    if max_eps > base_eps:
                        if eps <= 0.0:
                            eps = min(max_eps, max(base_eps, 0.25 * max_eps))
                        else:
                            eps = min(max_eps, max(base_eps, 1.25 * eps))

                self.pseudo_cfl_target_field[None] = cfl_target
                self.residual_smoothing_eps_field[None] = eps
                self.pseudo_reference_residual_field[None] = latest
            self.pseudo_cfl_eff_field[None] = float(self.pseudo_cfl_target_field[None])

    def _record_local_pseudo_residual(self, residual: float):
        """記錄當前 pseudo iteration 的 residual，供下一步自適應控制使用"""
        self.pseudo_latest_residual_field[None] = float(residual)

    def get_residual_norm(self) -> float:
        """
        回傳 interior cells 密度方程殘差 RMS（step() 後呼叫）

        What: L2 norm of R[i,j][0]（密度殘差）除以格點數
        Why:  收斂監控：穩態時 R → 0，此值應單調下降至機器精度附近
        When: 每次 step() 之後，R 仍保存最後 RK 段的空間殘差
        """
        return float(self._compute_residual_norm())

    def init_from_primitive_numpy(self, W_np: np.ndarray):
        """
        從 numpy 陣列設定 interior cells 的原始量，再更新 ghost

        What: 把任意分佈的 (ni, nj, 4) 陣列上傳至 solver field
        Why:  isentropic vortex 等非均勻 IC 無法用 init_uniform 設定

        Args:
            W_np: shape (ni, nj, 4) float32，分量順序 [ρ, u, v, p]
        """
        assert W_np.shape == (self.ni, self.nj, 4), \
            f"Expected ({self.ni}, {self.nj}, 4), got {W_np.shape}"

        NG = self.NG
        W_full = np.zeros((self.NI, self.NJ, 4), dtype=np.float32)
        W_full[NG:NG+self.ni, NG:NG+self.nj] = W_np.astype(np.float32)
        self.W.from_numpy(W_full)
        self._init_from_W_interior()
        self._update_ghost()

    # ─────────────────────────────────────────────────────────────
    # @ti.func：數值核心工具（在 kernel 內呼叫）
    # ─────────────────────────────────────────────────────────────

    @ti.func
    def _prim2cons(self, w: ti.types.vector(4, ti.f32)):
        """W = [ρ, u, v, p] → U = [ρ, ρu, ρv, E]"""
        rho, u, v, p = w[0], w[1], w[2], w[3]
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u*u + v*v)
        return ti.Vector([rho, rho*u, rho*v, E])

    @ti.func
    def _cons2prim(self, u_vec: ti.types.vector(4, ti.f32)):
        """U = [ρ, ρu, ρv, E] → W = [ρ, u, v, p]"""
        rho = u_vec[0]
        u   = u_vec[1] / rho
        v   = u_vec[2] / rho
        p   = (self.gamma - 1.0) * (u_vec[3] - 0.5 * rho * (u*u + v*v))
        return ti.Vector([rho, u, v, p])

    @ti.func
    def _euler_flux(self, w: ti.types.vector(4, ti.f32),
                    s: ti.types.vector(2, ti.f32)):
        """
        2D Euler 面法向通量 F·S

        What: 把原始量 W 投影到面向量 S，計算守恆通量
        Why:  用面向量 S（含方向與面積）而非 unit normal + 面積分開，
              避免 normalize 的浮點累積誤差
        """
        rho, u, v, p = w[0], w[1], w[2], w[3]
        E   = p / (self.gamma - 1.0) + 0.5 * rho * (u*u + v*v)
        un  = u * s[0] + v * s[1]   # 法向體積流量（= u_n · ΔS）
        return ti.Vector([
            rho * un,
            rho * u * un + p * s[0],
            rho * v * un + p * s[1],
            (E + p) * un,
        ])

    @ti.func
    def _minmod(self, a: ti.f32, b: ti.f32) -> ti.f32:
        """Minmod limiter（逐分量呼叫）"""
        r = 0.0
        if a * b > 0.0:
            r = ti.math.sign(a) * ti.min(ti.abs(a), ti.abs(b))
        return r

    @ti.func
    def _hllc(self, wL: ti.types.vector(4, ti.f32),
               wR: ti.types.vector(4, ti.f32),
               s:  ti.types.vector(2, ti.f32)):
        """
        2D HLLC Riemann Solver（面法向投影版）

        What: 把 2D 速度場投影到 face 法向，呼叫 1D HLLC，再還原
        Why:  直接在法向做 1D HLLC，避免張量收縮錯誤；
              切向速度透過 star state 直接傳遞（不影響壓力/ρ 跳）

        Args:
            wL, wR: face 左右原始量 [ρ, u, v, p]
            s:      面向量 S = n̂·ΔS（含方向與面積）
        """
        ds  = ti.math.length(s)           # 面積 ΔS
        nx  = s[0] / ds                   # unit normal x
        ny  = s[1] / ds                   # unit normal y

        rhoL, uL, vL, pL = wL[0], wL[1], wL[2], wL[3]
        rhoR, uR, vR, pR = wR[0], wR[1], wR[2], wR[3]

        # 法向速度
        unL = uL * nx + vL * ny
        unR = uR * nx + vR * ny

        EL  = pL / (self.gamma - 1.0) + 0.5 * rhoL * (uL*uL + vL*vL)
        ER  = pR / (self.gamma - 1.0) + 0.5 * rhoR * (uR*uR + vR*vR)

        cL  = ti.sqrt(self.gamma * pL / rhoL)
        cR  = ti.sqrt(self.gamma * pR / rhoR)

        # 波速估計（Davis）
        SL  = ti.min(unL - cL, unR - cR)
        SR  = ti.max(unL + cL, unR + cR)

        # 接觸面速度
        denom = rhoL*(SL - unL) - rhoR*(SR - unR)
        SM    = (pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR)) / denom

        # 守恆量與通量（face 法向方向，乘 ds 後是體積通量）
        UL_vec = ti.Vector([rhoL, rhoL*uL, rhoL*vL, EL])
        UR_vec = ti.Vector([rhoR, rhoR*uR, rhoR*vR, ER])
        FL     = self._euler_flux(wL, s)
        FR     = self._euler_flux(wR, s)

        # Star states（HLLC，Toro 1994 eq. 10.38 推廣至 2D）
        coefL = rhoL * (SL - unL) / (SL - SM)
        coefR = rhoR * (SR - unR) / (SR - SM)

        UstarL = coefL * ti.Vector([
            1.0,
            uL + (SM - unL) * nx,
            vL + (SM - unL) * ny,
            EL/rhoL + (SM - unL) * (SM + pL / (rhoL * (SL - unL))),
        ])
        UstarR = coefR * ti.Vector([
            1.0,
            uR + (SM - unR) * nx,
            vR + (SM - unR) * ny,
            ER/rhoR + (SM - unR) * (SM + pR / (rhoR * (SR - unR))),
        ])

        result = FR
        if SL >= 0.0:
            result = FL
        elif SM >= 0.0:
            result = FL + SL * ds * (UstarL - UL_vec)
        elif SR >= 0.0:
            result = FR + SR * ds * (UstarR - UR_vec)
        return result

    # ─────────────────────────────────────────────────────────────
    # Kernels
    # ─────────────────────────────────────────────────────────────

    @ti.kernel
    def _fill_cartesian_geometry(self, dx: ti.f32, dy: ti.f32):
        """直角網格：均勻面向量與控制體積"""
        for i, j in self.vol:
            # i-face(i+½, j)：法向 = +x，面積 = dy
            self.S_i[i, j] = ti.Vector([dy, 0.0])
            # j-face(i, j+½)：法向 = +y，面積 = dx
            self.S_j[i, j] = ti.Vector([0.0, dx])
            self.vol[i, j] = dx * dy

    @ti.kernel
    def _fill_curvilinear_geometry(
            self,
            xn: ti.types.ndarray(dtype=ti.f32, ndim=2),
            yn: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        """
        曲線結構網格：從節點座標計算面向量與控制體積

        面向量正向約定（右手法則，指向 cell 增加方向）：
          S_i[i,j] = i-face(i+½,j)：從 node(i+1,j) → node(i+1,j+1)，n̂ 指向 +i
            → S_i = ( dy_f, -dx_f )，dx_f = x[i+1,j+1]-x[i+1,j]
          S_j[i,j] = j-face(i,j+½)：從 node(i,j+1) → node(i+1,j+1)，n̂ 指向 +j
            → S_j = (-dy_g,  dx_g )，dx_g = x[i+1,j+1]-x[i,j+1]

        Cartesian 驗證：dx_f=0, dy_f=dy → S_i=(dy,0) ✓
                        dx_g=dx, dy_g=0 → S_j=(0,dx) ✓

        Cell 面積：quad 兩對角線叉積的一半（Shoelace 公式）
        """
        for i, j in self.vol:
            # i-face(i+½,j)
            dx_f = xn[i + 1, j + 1] - xn[i + 1, j]
            dy_f = yn[i + 1, j + 1] - yn[i + 1, j]
            self.S_i[i, j] = ti.Vector([dy_f, -dx_f])

            # j-face(i,j+½)
            dx_g = xn[i + 1, j + 1] - xn[i, j + 1]
            dy_g = yn[i + 1, j + 1] - yn[i, j + 1]
            self.S_j[i, j] = ti.Vector([-dy_g, dx_g])

            # Cell area（diagonals cross-product / 2）
            d1x = xn[i + 1, j + 1] - xn[i, j]
            d1y = yn[i + 1, j + 1] - yn[i, j]
            d2x = xn[i + 1, j    ] - xn[i, j + 1]
            d2y = yn[i + 1, j    ] - yn[i, j + 1]
            self.vol[i, j] = 0.5 * ti.abs(d1x * d2y - d1y * d2x)

    @ti.kernel
    def _init_uniform_kernel(self, rho: ti.f32, u: ti.f32,
                              v: ti.f32, p: ti.f32):
        """均勻初始化所有 cell（含 ghost）"""
        for i, j in self.U:
            self.W[i, j] = ti.Vector([rho, u, v, p])
            E = p / (self.gamma - 1.0) + 0.5 * rho * (u*u + v*v)
            self.U[i, j] = ti.Vector([rho, rho*u, rho*v, E])

    # ── 內部流程輔助（殘差評估、狀態切換）────────────────────────

    def _eval_residual(self):
        """reconstruct → flux → accumulate（用當前 W）"""
        self._reconstruct_faces()
        self._compute_fluxes()
        self._accumulate_residual()
        if self._time_marching_mode == "local_pseudo":
            self._smooth_conserved_residual()

    def _smooth_conserved_residual(self):
        """
        對守恆量殘差做顯式平滑

        What: 使用 5-point Jacobi-like filter 平滑高頻殘差
        Why:  只在 steady local pseudo-time 模式下注入少量人工耦合，
              不改動物理通量與 global physical-time marching
        """
        passes = int(self.residual_smoothing_passes_field[None])
        eps = float(self.residual_smoothing_eps_field[None])
        if passes <= 0 or eps <= 0.0:
            return
        for _ in range(passes):
            self._residual_smoothing_pass(eps)
            self._copy_smoothed_residual()

    def _assert_positive_state(self, src, stage_name: str):
        """
        在 RK stage 後檢查 interior conserved state 是否仍為物理解

        What: 驗證 rho > 0 且由 conserved variables 還原的 pressure > 0
        Why:  若負密度/負壓先流入下一輪 CFL 或 BC 計算，sqrt(gamma*p/rho)
              會先產生 NaN，錯誤位置就被洗掉；在 stage 邊界 fail fast
              才能把責任定位在真正出錯的時間步
        When: 每個 RK stage 更新完、切回當前狀態前
        """
        self._validate_conserved_state(src)
        bad_flat = int(self.bad_state_flat[None])
        sentinel = self.NI * self.NJ
        if bad_flat == sentinel:
            return

        i = bad_flat // self.NJ
        j = bad_flat % self.NJ
        u_vec = src.to_numpy()[i, j]
        rho = float(u_vec[0])
        if rho > 0.0 and np.isfinite(rho):
            kinetic = 0.5 * float(u_vec[1] * u_vec[1] + u_vec[2] * u_vec[2]) / rho
            p = float((self.gamma - 1.0) * (u_vec[3] - kinetic))
        else:
            p = float("nan")

        raise RuntimeError(
            f"Non-physical state after {stage_name}: "
            f"cell=({i - self.NG}, {j - self.NG}), "
            f"rho={rho:.6e}, p={p:.6e}"
        )

    @ti.kernel
    def _reconstruct_faces(self):
        """
        MUSCL face 重建（原始量空間 + 正性回退）

        i-face(i+½, j)：cell[i,j] → WL_i[i,j]，cell[i+1,j] → WR_i[i,j]
        j-face(i, j+½)：cell[i,j] → WL_j[i,j]，cell[i,j+1] → WR_j[i,j]

        Loop i: [NG-1, NG+ni)  →  訪問 W[i-1..i+2, j]，2 層 ghost 安全
        Loop j: [NG-1, NG+nj)  →  訪問 W[i, j-1..j+2]，2 層 ghost 安全
        """
        # ── i-faces ─────────────────────────────────────────────
        for i, j in ti.ndrange((self.NG - 1, self.NG + self.ni),
                                (self.NG,     self.NG + self.nj)):
            if self.first_order_recon_field[None] == 1:
                self.WL_i[i, j] = self.W[i, j]
                self.WR_i[i, j] = self.W[i + 1, j]
                continue

            # Left state（cell i）
            dWm = self.W[i,     j] - self.W[i - 1, j]
            dWp = self.W[i + 1, j] - self.W[i,     j]
            sL  = ti.Vector([self._minmod(dWm[k], dWp[k]) for k in ti.static(range(4))])
            wL_c = self.W[i, j] + 0.5 * sL

            # Right state（cell i+1）
            dWm2 = self.W[i + 1, j] - self.W[i,     j]
            dWp2 = self.W[i + 2, j] - self.W[i + 1, j]
            sR   = ti.Vector([self._minmod(dWm2[k], dWp2[k]) for k in ti.static(range(4))])
            wR_c = self.W[i + 1, j] - 0.5 * sR

            # 正性回退：ρ 或 p ≤ 0 → 一階
            if wL_c[0] <= 0.0 or wL_c[3] <= 0.0 or \
               wR_c[0] <= 0.0 or wR_c[3] <= 0.0:
                self.WL_i[i, j] = self.W[i,     j]
                self.WR_i[i, j] = self.W[i + 1, j]
            else:
                self.WL_i[i, j] = wL_c
                self.WR_i[i, j] = wR_c

        # ── j-faces ─────────────────────────────────────────────
        for i, j in ti.ndrange((self.NG,     self.NG + self.ni),
                                (self.NG - 1, self.NG + self.nj)):
            if self.first_order_recon_field[None] == 1:
                self.WL_j[i, j] = self.W[i, j]
                self.WR_j[i, j] = self.W[i, j + 1]
                continue

            # Left state（cell j）
            dWm = self.W[i, j    ] - self.W[i, j - 1]
            dWp = self.W[i, j + 1] - self.W[i, j    ]
            sL  = ti.Vector([self._minmod(dWm[k], dWp[k]) for k in ti.static(range(4))])
            wL_c = self.W[i, j] + 0.5 * sL

            # Right state（cell j+1）
            dWm2 = self.W[i, j + 1] - self.W[i, j    ]
            dWp2 = self.W[i, j + 2] - self.W[i, j + 1]
            sR   = ti.Vector([self._minmod(dWm2[k], dWp2[k]) for k in ti.static(range(4))])
            wR_c = self.W[i, j + 1] - 0.5 * sR

            if wL_c[0] <= 0.0 or wL_c[3] <= 0.0 or \
               wR_c[0] <= 0.0 or wR_c[3] <= 0.0:
                self.WL_j[i, j] = self.W[i, j    ]
                self.WR_j[i, j] = self.W[i, j + 1]
            else:
                self.WL_j[i, j] = wL_c
                self.WR_j[i, j] = wR_c

    @ti.kernel
    def _compute_fluxes(self):
        """i-faces 與 j-faces 各呼叫 HLLC"""
        for i, j in ti.ndrange((self.NG - 1, self.NG + self.ni),
                                (self.NG,     self.NG + self.nj)):
            self.Flux_i[i, j] = self._hllc(
                self.WL_i[i, j], self.WR_i[i, j], self.S_i[i, j])

        for i, j in ti.ndrange((self.NG,     self.NG + self.ni),
                                (self.NG - 1, self.NG + self.nj)):
            self.Flux_j[i, j] = self._hllc(
                self.WL_j[i, j], self.WR_j[i, j], self.S_j[i, j])

    @ti.kernel
    def _accumulate_residual(self):
        """
        R[i,j] = (Flux_i[i,j] - Flux_i[i-1,j]   ← i 方向面通量差
               +  Flux_j[i,j] - Flux_j[i,j-1])   ← j 方向面通量差
               / vol[i,j]

        cell[i,j] 的右面 i-flux = Flux_i[i,j]，左面 = Flux_i[i-1,j]
        cell[i,j] 的上面 j-flux = Flux_j[i,j]，下面 = Flux_j[i,j-1]
        """
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            self.R[i, j] = (
                (self.Flux_i[i, j] - self.Flux_i[i - 1, j] +
                 self.Flux_j[i, j] - self.Flux_j[i, j - 1])
                / self.vol[i, j]
            )

    def _set_state(self, src):
        """把 src 複製至 U，重新計算 W 和 ghost"""
        self._copy_to_U(src)
        self._update_primitive()
        self._update_ghost()

    @ti.kernel
    def _save_U0(self):
        for i, j in self.U:
            self.U0[i, j] = self.U[i, j]

    @ti.kernel
    def _copy_to_U(self, src: ti.template()):
        for i, j in self.U:
            self.U[i, j] = src[i, j]

    @ti.kernel
    def _validate_conserved_state(self, src: ti.template()):
        """
        檢查 interior conserved state 的正性

        What: 從 U=[rho, rho*u, rho*v, E] 直接檢查 rho 與 pressure 是否為正
        Why:  這是 solver 進入下一輪通量與邊界計算前的最低物理門檻
        """
        sentinel = self.NI * self.NJ
        self.bad_state_flat[None] = sentinel
        rho_floor = 1e-8
        p_floor = 1e-8

        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho = src[i, j][0]
            rhou = src[i, j][1]
            rhov = src[i, j][2]
            E = src[i, j][3]

            invalid = 0
            if not (rho > rho_floor):
                invalid = 1
            else:
                kinetic = 0.5 * (rhou * rhou + rhov * rhov) / rho
                p = (self.gamma - 1.0) * (E - kinetic)
                if not (p > p_floor):
                    invalid = 1

            if invalid == 1:
                ti.atomic_min(self.bad_state_flat[None], i * self.NJ + j)

    @ti.kernel
    def _update_primitive(self):
        """
        Interior cells：U = [ρ, ρu, ρv, E] → W = [ρ, u, v, p]

        Why: limiter 與重建在原始量空間操作，避免重建後出現非物理狀態
        """
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho  = self.U[i, j][0]
            rhou = self.U[i, j][1]
            rhov = self.U[i, j][2]
            E    = self.U[i, j][3]
            u_   = rhou / rho
            v_   = rhov / rho
            p    = (self.gamma - 1.0) * (E - 0.5 * rho * (u_*u_ + v_*v_))
            self.W[i, j] = ti.Vector([rho, u_, v_, p])

    def _update_ghost(self):
        """
        Ghost cells 邊界條件分派器

        各方向獨立設定，支援混合 BC：
          i：periodic / nozzle characteristic / Neumann
          j_min：slip_wall / periodic / Neumann
          j_max：periodic / Neumann
        """
        # ── i 方向 ─────────────────────────────────────
        if self._periodic_i:
            self._apply_periodic_i_bc()
        else:
            if self._nozzle_inlet_i_min:
                self._apply_nozzle_inlet_i_min(
                    float(self._nozzle_p0),
                    float(self._nozzle_t0),
                    float(self._nozzle_mach_in),
                    float(self._nozzle_inlet_relax),
                )
            else:
                self._apply_neumann_i_min_bc()
            if self._nozzle_outlet_i_max:
                self._apply_nozzle_outlet_i_max(
                    float(self._nozzle_back_pressure),
                    float(self._nozzle_outlet_relax),
                )
            else:
                self._apply_neumann_i_max_bc()

        # ── j_min（內壁 / 下壁）────────────────────────
        if self._wall_j_min:
            self._apply_slip_wall_j_min()
        elif self._periodic_j:
            self._apply_periodic_j_min_bc()
        else:
            self._apply_neumann_j_min_bc()

        # ── j_max（外壁 / 遠場 / 上壁）─────────────────
        if self._wall_j_max:
            self._apply_slip_wall_j_max()
        elif self._far_field_j_max:
            self._apply_far_field_j_max(
                float(self._rho_inf), float(self._u_inf),
                float(self._v_inf),   float(self._p_inf))
        elif self._periodic_j:
            self._apply_periodic_j_max_bc()
        else:
            self._apply_neumann_j_max_bc()

    # ── i-direction BCs ──────────────────────────────────────────

    @ti.kernel
    def _apply_neumann_i_min_bc(self):
        """i_min Neumann：left ghost ← 最左 interior"""
        for g, j in ti.ndrange(self.NG, self.NJ):
            self.W[g, j] = self.W[self.NG, j]
            self.U[g, j] = self.U[self.NG, j]

    @ti.kernel
    def _apply_neumann_i_max_bc(self):
        """i_max Neumann：right ghost ← 最右 interior"""
        ri = self.NG + self.ni - 1
        for g, j in ti.ndrange(self.NG, self.NJ):
            self.W[self.NG + self.ni + g, j] = self.W[ri, j]
            self.U[self.NG + self.ni + g, j] = self.U[ri, j]

    @ti.kernel
    def _apply_periodic_i_bc(self):
        """i 方向周期：left ghost ← right interior，right ghost ← left interior

        ghost_left[g]    ← interior[ni+g]   (右側 NG 個 interior cells)
        ghost_right[NG+ni+g] ← interior[NG+g]  (左側 NG 個 interior cells)
        """
        for g, j in ti.ndrange(self.NG, self.NJ):
            self.W[g, j] = self.W[self.ni + g, j]
            self.U[g, j] = self.U[self.ni + g, j]
            self.W[self.NG + self.ni + g, j] = self.W[self.NG + g, j]
            self.U[self.NG + self.ni + g, j] = self.U[self.NG + g, j]

    @ti.kernel
    def _apply_nozzle_inlet_i_min(
            self, p0: ti.f32, t0: ti.f32, mach_in: ti.f32, relax: ti.f32):
        """
        i_min nozzle 入口特徵 BC（總壓/總溫）

        What: 亞音速入口以 interior 出射特徵 R- + reservoir 總量 p0/T0 重建；
              若局部轉為超音速入口，退回由 reservoir 全指定（supersonic inflow）
        Why:  p0/T0 是內流 nozzle 的真正控制量，避免額外固定 Mach 過度約束
        """
        eps = 1e-8
        gamma = self.gamma
        gm1 = gamma - 1.0
        m_target = ti.min(2.5, ti.max(1e-4, mach_in))

        fac_t = 1.0 + 0.5 * gm1 * m_target * m_target
        t_target = ti.max(t0 / fac_t, 1e-6)
        c_target = ti.sqrt(gamma * t_target)
        u_target = m_target * c_target
        r_plus_target = u_target + 2.0 * c_target / gm1

        for g, j in ti.ndrange(self.NG, (self.NG, self.NG + self.nj)):
            i_ghost = self.NG - 1 - g
            i_int = self.NG + g

            rho_i = ti.max(self.W[i_int, j][0], 1e-6)
            u_i = self.W[i_int, j][1]
            v_i = self.W[i_int, j][2]
            p_i = ti.max(self.W[i_int, j][3], 1e-6)

            c_i = ti.sqrt(gamma * p_i / rho_i)

            si = self.S_i[self.NG - 1, j]
            ds = ti.max(ti.math.length(si), eps)
            nx = si[0] / ds
            ny = si[1] / ds
            tx = -ny
            ty = nx

            u_n_i = u_i * nx + v_i * ny
            u_t_i = u_i * tx + v_i * ty
            u_n_char = u_target
            c_b = c_target
            p_char = p_i
            rho_char = rho_i

            # 亞音速入口：R- 由 interior 決定，配合 t0 封閉問題得到 (u_n, c)
            # 超音速入口：全部特徵由 reservoir 指定，不應再使用 interior R-
            is_supersonic_in = u_n_i >= c_i
            if (m_target < 1.0) and (not is_supersonic_in):
                r_minus_i = u_n_i - 2.0 * c_i / gm1
                k = 2.0 / gm1
                # u_n^2 + 2/(γ-1) c^2 = 2γ/(γ-1) t0, 且 u_n = r_minus + k c
                a = 2.0 * gamma * t0 / gm1
                qa = k * k + 2.0 / gm1
                qb = 2.0 * r_minus_i * k
                qc = r_minus_i * r_minus_i - a
                disc = ti.max(qb * qb - 4.0 * qa * qc, 0.0)
                sqrt_disc = ti.sqrt(disc)
                c1 = (-qb + sqrt_disc) / (2.0 * qa)
                c2 = (-qb - sqrt_disc) / (2.0 * qa)
                c_b = ti.max(1e-4, ti.max(c1, c2))
                u_n_char = r_minus_i + k * c_b
            # 超音速入口：全部特徵由 reservoir 指定，退回目標 Mach 模式
            else:
                c_b = c_target
                u_n_char = u_target

            u_n_char = ti.max(u_n_char, 1e-6)
            t_b = ti.max(c_b * c_b / gamma, 1e-6)
            tau = ti.min(1.0, ti.max(t_b / ti.max(t0, 1e-6), 1e-6))
            p_char = ti.max(p0 * ti.pow(tau, gamma / gm1), 1e-6)
            rho_char = ti.max(p_char / t_b, 1e-6)

            rho_b = (1.0 - relax) * rho_i + relax * rho_char
            p_b = (1.0 - relax) * p_i + relax * p_char
            u_n_b = (1.0 - relax) * u_n_i + relax * u_n_char
            rho_b = ti.max(rho_b, 1e-6)
            p_b = ti.max(p_b, 1e-6)

            u_b = u_n_b * nx + u_t_i * tx
            v_b = u_n_b * ny + u_t_i * ty
            self.W[i_ghost, j] = ti.Vector([rho_b, u_b, v_b, p_b])
            self.U[i_ghost, j] = self._prim2cons(self.W[i_ghost, j])

    @ti.kernel
    def _apply_nozzle_outlet_i_max(self, back_pressure: ti.f32, relax: ti.f32):
        """
        i_max nozzle 出口特徵 BC（背壓）

        What: 亞音速出口以 R+ (interior) + 指定 pb 重建；超音速出口採外插
        Why:  內流 nozzle 的壓力比控制點在出口背壓
        """
        eps = 1e-8
        gamma = self.gamma
        gm1 = gamma - 1.0
        pb = ti.max(back_pressure, 1e-6)
        for g, j in ti.ndrange(self.NG, (self.NG, self.NG + self.nj)):
            i_ghost = self.NG + self.ni + g
            i_int = self.NG + self.ni - 1 - g

            rho_i = ti.max(self.W[i_int, j][0], 1e-6)
            u_i = self.W[i_int, j][1]
            v_i = self.W[i_int, j][2]
            p_i = ti.max(self.W[i_int, j][3], 1e-6)

            c_i = ti.sqrt(gamma * p_i / rho_i)
            si = self.S_i[self.NG + self.ni - 1, j]
            ds = ti.max(ti.math.length(si), eps)
            nx = si[0] / ds
            ny = si[1] / ds
            tx = -ny
            ty = nx

            u_n_i = u_i * nx + v_i * ny
            u_t_i = u_i * tx + v_i * ty

            is_supersonic_out = u_n_i >= c_i

            r_plus_i = u_n_i + 2.0 * c_i / gm1
            s_i = p_i / ti.pow(rho_i, gamma)
            s_i = ti.max(s_i, 1e-8)
            rho_pb = ti.pow(pb / s_i, 1.0 / gamma)
            rho_pb = ti.max(rho_pb, 1e-6)
            c_pb = ti.sqrt(gamma * pb / rho_pb)
            u_n_pb = r_plus_i - 2.0 * c_pb / gm1

            rho_bc = ti.select(is_supersonic_out, rho_i, rho_pb)
            p_bc = ti.select(is_supersonic_out, p_i, pb)
            u_n_bc = ti.select(is_supersonic_out, u_n_i, u_n_pb)

            rho_b = (1.0 - relax) * rho_i + relax * rho_bc
            # Subsonic outlet 的控制量是背壓 pb，直接強制施加；
            # Supersonic outlet 則保持外插。
            p_b = ti.select(is_supersonic_out, p_i, p_bc)
            u_n_b = (1.0 - relax) * u_n_i + relax * u_n_bc
            rho_b = ti.max(rho_b, 1e-6)
            p_b = ti.max(p_b, 1e-6)

            u_b = u_n_b * nx + u_t_i * tx
            v_b = u_n_b * ny + u_t_i * ty
            self.W[i_ghost, j] = ti.Vector([rho_b, u_b, v_b, p_b])
            self.U[i_ghost, j] = self._prim2cons(self.W[i_ghost, j])

    # ── j_min BCs ────────────────────────────────────────────────

    @ti.kernel
    def _apply_neumann_j_min_bc(self):
        """j_min Neumann：bottom ghost ← 最近 interior"""
        for i, g in ti.ndrange(self.NI, self.NG):
            self.W[i, g] = self.W[i, self.NG]
            self.U[i, g] = self.U[i, self.NG]

    @ti.kernel
    def _apply_periodic_j_min_bc(self):
        """j_min 周期：bottom ghost ← top interior"""
        for i, g in ti.ndrange(self.NI, self.NG):
            self.W[i, g] = self.W[i, self.nj + g]
            self.U[i, g] = self.U[i, self.nj + g]

    @ti.kernel
    def _apply_slip_wall_j_min(self):
        """
        j_min 無黏滑動壁 BC（Euler 翼型表面）

        What: ghost cell = interior cell 以壁面法線鏡像後的狀態
              (ρ, p 不變，法向速度反號，切向速度保留)
        Why:  等效於牆面不可穿透條件 v_n = 0，讓 HLLC 對壁面 face 得到正確通量
        How:  壁面法向量取 S_j[i, NG-1]（壁面 j-face），對每層 ghost 使用相同法向
        """
        for i, g in ti.ndrange(self.NI, self.NG):
            j_int   = self.NG + g        # interior: NG, NG+1
            j_ghost = self.NG - 1 - g   # ghost:    NG-1, NG-2

            rho = self.W[i, j_int][0]
            u   = self.W[i, j_int][1]
            v   = self.W[i, j_int][2]
            p   = self.W[i, j_int][3]

            # 壁面法向量（始終用壁面 face，即 NG-1 層 j-face）
            sj = self.S_j[i, self.NG - 1]
            ds = ti.math.length(sj)
            nx = sj[0] / ds
            ny = sj[1] / ds

            # 反射法向速度
            un    = u * nx + v * ny
            u_g   = u - 2.0 * un * nx
            v_g   = v - 2.0 * un * ny

            self.W[i, j_ghost] = ti.Vector([rho, u_g, v_g, p])
            self.U[i, j_ghost] = self._prim2cons(self.W[i, j_ghost])

    # ── j_max BCs ────────────────────────────────────────────────

    @ti.kernel
    def _apply_neumann_j_max_bc(self):
        """j_max Neumann：top ghost ← 最近 interior"""
        tj = self.NG + self.nj - 1
        for i, g in ti.ndrange(self.NI, self.NG):
            self.W[i, self.NG + self.nj + g] = self.W[i, tj]
            self.U[i, self.NG + self.nj + g] = self.U[i, tj]

    @ti.kernel
    def _apply_periodic_j_max_bc(self):
        """j_max 周期：top ghost ← bottom interior"""
        for i, g in ti.ndrange(self.NI, self.NG):
            self.W[i, self.NG + self.nj + g] = self.W[i, self.NG + g]
            self.U[i, self.NG + self.nj + g] = self.U[i, self.NG + g]

    @ti.kernel
    def _apply_far_field_j_max(self,
                                rho_inf: ti.f32, u_inf: ti.f32,
                                v_inf: ti.f32,   p_inf: ti.f32):
        """
        j_max 特徵遠場 BC（Riemann invariant，適用亞音速）

        物理基礎：1D Riemann 不變量沿法向傳播
          R⁺ = u_n + 2c/(γ-1)   wave speed λ⁺ = u_n + c
          R⁻ = u_n - 2c/(γ-1)   wave speed λ⁻ = u_n - c
          s  = p/ρ^γ             wave speed λ⁰ = u_n（熵波）
          u_t                    wave speed λ⁰（切向，被動輸送）

        亞音速遠場（|u_n| < c）：
          λ⁺ > 0 → R⁺ 從 interior 出射（取 interior 值）
          λ⁻ < 0 → R⁻ 從 freestream 入射（取自由流值）
          λ⁰ 依 u_n 正負判斷 s, u_t 來源

        重建公式：
          u_n_b = (R⁺ + R⁻) / 2
          c_b   = (R⁺ - R⁻) * (γ-1) / 4
          ρ_b   = (c_b² / (γ s_b))^{1/(γ-1)}
          p_b   = s_b * ρ_b^γ
        """
        c_inf = ti.sqrt(self.gamma * p_inf / rho_inf)
        s_inf = p_inf / ti.pow(rho_inf, self.gamma)
        tj    = self.NG + self.nj - 1   # last interior j-index

        for i, g in ti.ndrange(self.NI, self.NG):
            j_ghost = self.NG + self.nj + g   # ghost index

            # Interior 狀態（取最後一層 interior cell）
            rho_i = self.W[i, tj][0]
            u_i   = self.W[i, tj][1]
            v_i   = self.W[i, tj][2]
            p_i   = self.W[i, tj][3]
            c_i   = ti.sqrt(self.gamma * p_i / rho_i)
            s_i   = p_i / ti.pow(rho_i, self.gamma)

            # j_max 面法向（指向 +j = away from airfoil = outward）
            sj = self.S_j[i, tj]
            ds = ti.math.length(sj)
            nx = sj[0] / ds
            ny = sj[1] / ds
            # 切向（右手 90° 旋轉：t = (-ny, nx)）
            tx = -ny;  ty = nx

            # 投影法向速度
            u_n_i   = u_i   * nx + v_i   * ny
            u_t_i   = u_i   * tx + v_i   * ty
            u_n_inf = u_inf * nx + v_inf * ny
            u_t_inf = u_inf * tx + v_inf * ty

            # Riemann 不變量
            R_plus_i    = u_n_i   + 2.0 * c_i   / (self.gamma - 1.0)
            R_minus_inf_ = u_n_inf - 2.0 * c_inf / (self.gamma - 1.0)

            # 亞音速：λ⁻ < 0 → R⁻ 取自由流；超音速出流保護（罕見）
            R_minus_i_  = u_n_i - 2.0 * c_i / (self.gamma - 1.0)
            R_minus_b   = ti.select(u_n_i - c_i < 0.0, R_minus_inf_, R_minus_i_)

            # 重建邊界法向速度與音速
            u_n_b = 0.5 * (R_plus_i + R_minus_b)
            c_b   = 0.25 * (R_plus_i - R_minus_b) * (self.gamma - 1.0)
            c_b   = ti.max(c_b, 1e-4)   # 防止 c_b ≤ 0

            # 熵與切向速度：由 u_n 符號決定來源（ti.select 避免 SSA 限制）
            is_outflow = u_n_i >= 0.0
            s_b   = ti.select(is_outflow, s_i,   s_inf)
            u_t_b = ti.select(is_outflow, u_t_i, u_t_inf)
            s_b   = ti.max(s_b, 1e-8)

            # 重建 ρ, p（從 c_b 和 s_b = p/ρ^γ）
            exp   = 1.0 / (self.gamma - 1.0)
            rho_b = ti.pow(c_b * c_b / (self.gamma * s_b), exp)
            p_b   = s_b * ti.pow(rho_b, self.gamma)
            rho_b = ti.max(rho_b, 1e-6)
            p_b   = ti.max(p_b,   1e-6)

            # 重建 2D 速度
            u_b = u_n_b * nx + u_t_b * tx
            v_b = u_n_b * ny + u_t_b * ty

            self.W[i, j_ghost] = ti.Vector([rho_b, u_b, v_b, p_b])
            self.U[i, j_ghost] = self._prim2cons(self.W[i, j_ghost])

    @ti.kernel
    def _apply_slip_wall_j_max(self):
        """j_max 無黏滑動壁 BC（通道上壁）"""
        for i, g in ti.ndrange(self.NI, self.NG):
            j_int   = self.NG + self.nj - 1 - g
            j_ghost = self.NG + self.nj + g

            rho = self.W[i, j_int][0]
            u   = self.W[i, j_int][1]
            v   = self.W[i, j_int][2]
            p   = self.W[i, j_int][3]

            sj = self.S_j[i, self.NG + self.nj - 1]
            ds = ti.math.length(sj)
            nx = sj[0] / ds
            ny = sj[1] / ds

            un  = u * nx + v * ny
            u_g = u - 2.0 * un * nx
            v_g = v - 2.0 * un * ny

            self.W[i, j_ghost] = ti.Vector([rho, u_g, v_g, p])
            self.U[i, j_ghost] = self._prim2cons(self.W[i, j_ghost])

    @ti.kernel
    def _compute_residual_norm(self) -> ti.f32:
        """
        Interior cells 密度殘差 RMS

        Why density only: 量綱單純（1/time），不混合動量/能量的不同尺度；
                          密度收斂代表整個系統趨於穩態
        """
        s = 0.0
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            s += self.R[i, j][0] ** 2
        return ti.sqrt(s / (self.ni * self.nj))

    @ti.kernel
    def _residual_smoothing_pass(self, eps: ti.f32):
        """
        What: R_tmp = (R + eps * Σ_neighbors R) / (1 + 4 eps)
        Why:  Jacobi 形式可平行，避免 Gauss-Seidel 的資料相依
        """
        denom = 1.0 + 4.0 * eps
        i_min = self.NG
        i_max = self.NG + self.ni - 1
        j_min = self.NG
        j_max = self.NG + self.nj - 1
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            im = ti.max(i - 1, i_min)
            ip = ti.min(i + 1, i_max)
            jm = ti.max(j - 1, j_min)
            jp = ti.min(j + 1, j_max)
            self.R_tmp[i, j] = (
                self.R[i, j]
                + eps * (
                    self.R[im, j]
                    + self.R[ip, j]
                    + self.R[i, jm]
                    + self.R[i, jp]
                )
            ) / denom

    @ti.kernel
    def _copy_smoothed_residual(self):
        """把 smoothing 暫存結果覆寫回主殘差場"""
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            self.R[i, j] = self.R_tmp[i, j]

    @ti.kernel
    def _init_from_W_interior(self):
        """把 W[interior] 轉換成對應的 U[interior]（不動 ghost）"""
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            self.U[i, j] = self._prim2cons(self.W[i, j])

    @ti.kernel
    def _compute_dt(self):
        """
        全域最小時間步（顯式 Euler CFL）

        Why: 每個 cell 用自己的 λ_max / cell_size 算局部 dt，
             取全域最小值保證整域穩定
        Formula: dt = CFL * vol / Σ_faces (|u_n| + c) * |S_f|
        """
        self.dt_field[None] = 1.0e10
        self.dt_min_field[None] = 1.0e10
        self.dt_max_field[None] = 0.0
        self.dt_sum_field[None] = 0.0
        if self.time_marching_mode_field[None] == 1:
            self.pseudo_precond_scale_min_field[None] = 1.0e10
            self.pseudo_precond_scale_max_field[None] = 0.0
            self.pseudo_precond_scale_sum_field[None] = 0.0
            if self.pseudo_adaptive_enabled_field[None] == 1:
                self.pseudo_cfl_eff_field[None] = self.pseudo_cfl_target_field[None]
            else:
                cfl_start = self.pseudo_cfl_start_field[None]
                ramp_steps = self.pseudo_cfl_ramp_steps_field[None]
                iter_count = self.pseudo_iter_field[None]
                if ramp_steps > 0:
                    ramp_ratio = ti.min(1.0, ti.cast(iter_count, ti.f32) / ti.cast(ramp_steps, ti.f32))
                    self.pseudo_cfl_eff_field[None] = cfl_start + ramp_ratio * (self.cfl - cfl_start)
                else:
                    self.pseudo_cfl_eff_field[None] = self.cfl
        else:
            self.pseudo_cfl_eff_field[None] = self.cfl
            self.pseudo_precond_scale_min_field[None] = 1.0
            self.pseudo_precond_scale_max_field[None] = 1.0
            self.pseudo_precond_scale_sum_field[None] = 0.0
        for i, j in self.dt_local:
            self.dt_local[i, j] = 0.0
        for i, j in ti.ndrange((self.NG, self.NG + self.ni),
                                (self.NG, self.NG + self.nj)):
            rho, u, v, p = (self.W[i, j][0], self.W[i, j][1],
                            self.W[i, j][2], self.W[i, j][3])
            c   = ti.sqrt(self.gamma * p / rho)
            c_eff = c
            scale = 1.0
            if self.time_marching_mode_field[None] == 1:
                vel_mag = ti.sqrt(u * u + v * v)
                mach_local = vel_mag / ti.max(c, 1e-12)
                ref_mach = ti.max(self.pseudo_precond_ref_mach_field[None], 1e-6)
                min_scale = self.pseudo_precond_min_scale_field[None]
                scale = ti.max(min_scale, ti.min(1.0, mach_local / ref_mach))
                c_eff = c * scale
                ti.atomic_min(self.pseudo_precond_scale_min_field[None], scale)
                ti.atomic_max(self.pseudo_precond_scale_max_field[None], scale)
                ti.atomic_add(self.pseudo_precond_scale_sum_field[None], scale)

            # 四個 face 的波速貢獻（|u_n| + c）* |S_f|
            lam = 0.0
            # i-faces：左 S_i[i-1,j]，右 S_i[i,j]
            for di in ti.static(range(2)):
                si = self.S_i[i - 1 + di, j]
                ds = ti.math.length(si)
                nx = si[0] / ds;  ny = si[1] / ds
                un = ti.abs(u * nx + v * ny)
                lam += (un + c_eff) * ds
            # j-faces：下 S_j[i,j-1]，上 S_j[i,j]
            for dj in ti.static(range(2)):
                sj = self.S_j[i, j - 1 + dj]
                ds = ti.math.length(sj)
                nx = sj[0] / ds;  ny = sj[1] / ds
                un = ti.abs(u * nx + v * ny)
                lam += (un + c_eff) * ds

            dt_local = self.pseudo_cfl_eff_field[None] * self.vol[i, j] / lam
            self.dt_local[i, j] = dt_local
            ti.atomic_min(self.dt_field[None], dt_local)
            ti.atomic_min(self.dt_min_field[None], dt_local)
            ti.atomic_max(self.dt_max_field[None], dt_local)
            ti.atomic_add(self.dt_sum_field[None], dt_local)

    @ti.kernel
    def _rk_stage1(self, dt: ti.f32):
        """U1 = U0 - dt·R"""
        for i, j in self.U1:
            self.U1[i, j] = self.U0[i, j] - dt * self.R[i, j]

    @ti.kernel
    def _rk_stage1_local(self):
        """U1 = U0 - dt_local·R"""
        for i, j in self.U1:
            self.U1[i, j] = self.U0[i, j] - self.dt_local[i, j] * self.R[i, j]

    @ti.kernel
    def _rk_stage2(self, dt: ti.f32):
        """U2 = ¾ U0 + ¼ U1 - ¼ dt·R(U1)"""
        for i, j in self.U2:
            self.U2[i, j] = (0.75 * self.U0[i, j]
                             + 0.25 * self.U1[i, j]
                             - 0.25 * dt * self.R[i, j])

    @ti.kernel
    def _rk_stage2_local(self):
        """U2 = ¾ U0 + ¼ U1 - ¼ dt_local·R(U1)"""
        for i, j in self.U2:
            self.U2[i, j] = (0.75 * self.U0[i, j]
                             + 0.25 * self.U1[i, j]
                             - 0.25 * self.dt_local[i, j] * self.R[i, j])

    @ti.kernel
    def _rk_stage3(self, dt: ti.f32):
        """U = ⅓ U0 + ⅔ U2 - ⅔ dt·R(U2)"""
        for i, j in self.U:
            self.U[i, j] = ((1.0/3.0) * self.U0[i, j]
                            + (2.0/3.0) * self.U2[i, j]
                            - (2.0/3.0) * dt * self.R[i, j])

    @ti.kernel
    def _rk_stage3_local(self):
        """U = ⅓ U0 + ⅔ U2 - ⅔ dt_local·R(U2)"""
        for i, j in self.U:
            self.U[i, j] = ((1.0/3.0) * self.U0[i, j]
                            + (2.0/3.0) * self.U2[i, j]
                            - (2.0/3.0) * self.dt_local[i, j] * self.R[i, j])
