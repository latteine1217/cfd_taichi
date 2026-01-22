以下是針對我們目前的 NACA 2412 機翼模型，建議的參數設定指令：

  1. ✈️ 巡航模式 (Cruise Configuration)
  這是飛機在高空飛行時的狀態。追求高效率 (High L/D)，阻力最小化。
   * 特徵：所有襟翼收起，攻角很小，流線非常平滑，尾流很細。
   * 參數：
       * AoA: 2~4度
       * Slat: 0度
       * Flap: 0度

   1 uv run python run_airfoil.py --naca 2412 --aoa 3 --slat 0 --flap 0 --re 1000 --u_in 0.04 --steps 5000 --out output_cruise

  2. 🛫 起飛模式 (Takeoff Configuration)
  需要大量升力來讓飛機離開地面，但不能有太大的阻力。
   * 特徵：縫翼與襟翼適度展開，增加翼型彎度 (Camber)。您會看到氣流轉彎幅度變大，升力顯著上升。
   * 參數：
       * AoA: 10~12度 (抬頭仰角)
       * Slat: 15度 (中等偏轉)
       * Flap: 15度 (中等偏轉)

   1 uv run python run_airfoil.py --naca 2412 --aoa 10 --slat 15 --flap 15 --re 1000 --u_in 0.04 --steps 5000 --out output_takeoff

  3. 🛬 降落模式 (Landing Configuration)
  需要最大升力（以極低速飛行）和高阻力（幫助減速）。
   * 特徵：襟翼全開，就像一面空氣牆。這是一個非常劇烈的流場，您可能會在襟翼後方看到一些分離或強烈的下洗氣流 (Downwash)。這是展示 "High-Lift Device" 物理機制的最佳場景。
   * 參數：
       * AoA: 5~8度 (進場姿勢)
       * Slat: 20度 (全開，保護前緣)
       * Flap: 35~40度 (全開，最大阻力)

   1 uv run python run_airfoil.py --naca 2412 --aoa 6 --slat 20 --flap 40 --re 1000 --u_in 0.04 --steps 5000 --out output_landing

  4. ⚠️ 失速模式 (Stall / Deep Stall)
  展示當攻角過大時，氣流如何完全剝離機翼表面。
   * 特徵：機翼上方出現巨大的漩渦（Karman Vortex Street），升力驟降，阻力暴增。這展示了為什麼飛機需要縫翼來延後失速。
   * 參數：
       * AoA: 20~25度 (超臨界攻角)
       * Slat: 0度 (故意不開縫翼，讓它更容易失速)
       * Flap: 0度

   1 uv run python run_airfoil.py --naca 2412 --aoa 25 --slat 0 --flap 0 --re 1000 --u_in 0.04 --steps 5000 --out output_stall

  📊 觀察重點
  執行完上述模擬後，您可以對比 force_history.png：
   1. Cruise: $C_d$ 非常低，曲線平穩。
   2. Takeoff: $C_l$ 很高，效率 ($L/D$) 通常不錯。
   3. Landing: $C_l$ 最高，但 $C_d$ 也非常巨大（效率反而低，但這是降落需要的）。
   4. Stall: 曲線會劇烈震盪，代表非定常的渦旋脫落。
