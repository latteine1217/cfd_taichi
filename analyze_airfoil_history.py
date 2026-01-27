import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_history(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load data
    raw_data = np.load(file_path, allow_pickle=True).item()
    headers = raw_data['headers']
    data = np.array(raw_data['data'])
    
    # Extract columns
    # ['step', 'R_u', 'R_v', 'R_rho', 'M_err', 'Cd', 'Cl', 'Umax', 'ETA']
    steps = data[:, 0].astype(float)
    r_u = data[:, 1].astype(float)
    r_v = data[:, 2].astype(float)
    r_rho = data[:, 3].astype(float)
    m_err = data[:, 4].astype(float)
    cd = data[:, 5].astype(float)
    cl = data[:, 6].astype(float)
    
    # 1. Plot Residuals and Mass Error
    plt.figure(figsize=(10, 6))
    plt.semilogy(steps, r_u, label='R_u (Momentum X)')
    plt.semilogy(steps, r_v, label='R_v (Momentum Y)')
    plt.semilogy(steps, r_rho, label='R_rho (Density)')
    plt.semilogy(steps, m_err, label='Mass Error', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Error / Residual')
    plt.title('Convergence History (Residuals)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'))
    plt.close()
    
    # 2. Plot Aerodynamic Coefficients
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Calculate L/D ratio (handle division by zero if necessary, though Cd usually > 0)
    # Use a small epsilon to avoid division by zero just in case
    epsilon = 1e-10
    ld_ratio = cl / (cd + epsilon)

    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Coefficient (Cl, Cd)', color=color)
    l1 = ax1.plot(steps, cd, label='Cd (Drag)', color='tab:green', linestyle='--')
    l2 = ax1.plot(steps, cl, label='Cl (Lift)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.5)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('L/D Ratio', color=color)  
    l3 = ax2.plot(steps, ld_ratio, label='L/D', color=color, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combine legends
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')

    plt.title('Aerodynamic Coefficients & Efficiency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aerodynamics.png'))
    plt.close()
    
    print(f"Analysis complete. Plots saved to {output_dir}")
    print(f"Final Cd: {cd[-1]:.4f}")
    print(f"Final Cl: {cl[-1]:.4f}")
    print(f"Final L/D: {ld_ratio[-1]:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="分析機翼模擬歷史數據並生成收斂圖表"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output_airfoil',
        help='輸入目錄路徑（包含 history.npy 的目錄，預設: output_airfoil）'
    )
    parser.add_argument(
        '--output-subdir',
        type=str,
        default='analysis',
        help='輸出子目錄名稱（相對於輸入目錄，預設: analysis）'
    )

    args = parser.parse_args()

    # 構建完整路徑
    history_file = os.path.join(args.input_dir, 'history.npy')
    output_dir = os.path.join(args.input_dir, args.output_subdir)

    # 檢查輸入檔案是否存在
    if not os.path.exists(history_file):
        print(f"❌ 錯誤: 找不到歷史檔案 {history_file}")
        print(f"   請確認目錄 '{args.input_dir}' 中包含 history.npy")
        exit(1)

    print(f"📂 讀取歷史數據: {history_file}")
    print(f"📊 輸出目錄: {output_dir}\n")

    analyze_history(history_file, output_dir)
