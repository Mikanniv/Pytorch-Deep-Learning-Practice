import torch

def test1():
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

import time

def test2():
    # 检查 MPS 是否可用
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")

    # 测试规模（越大越能体现性能差距）
    size = 8000

    # 创建随机矩阵
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)

    # ===== CPU 测试 =====
    start = time.time()
    z_cpu = torch.mm(x_cpu, y_cpu)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    cpu_time = end - start
    print(f"CPU time: {cpu_time:.4f} seconds")

    # ===== MPS 测试 =====
    if mps_available:
        device = torch.device("mps")
        x_mps = x_cpu.to(device)
        y_mps = y_cpu.to(device)

        torch.mps.synchronize()  # 清空缓冲区
        start = time.time()
        z_mps = torch.mm(x_mps, y_mps)
        torch.mps.synchronize()  # 等待所有计算完成
        end = time.time()
        mps_time = end - start
        print(f"MPS time: {mps_time:.4f} seconds")

        # 验证结果是否一致
        z_mps_cpu = z_mps.to("cpu")
        diff = torch.abs(z_cpu - z_mps_cpu).max().item()
        print(f"Max difference between CPU and MPS: {diff:.6f}")

        print(f"🚀 Speedup: {cpu_time/mps_time:.2f}x faster on MPS")
    else:
        print("⚠️ MPS backend not available on this device.")


import gymnasium as gym

def run_env(env_name, steps=500):
    print(f"\n=== Testing {env_name} ===")
    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset()
    print("Initial observation:", obs)

    for _ in range(steps):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()

        time.sleep(0.02)  # 控制速度

    env.close()
    print(f"=== {env_name} test finished ===")


if __name__ == "__main__":
    # 测试 CartPole
    run_env("CartPole-v1", steps=300)

    # 测试 LunarLander (Box2D 环境)
    run_env("LunarLander-v3", steps=500)