################ 完整代码
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
def show(env,state_size, action_size,model_path):
    # 初始化环境和模型
    model = Network(state_size, action_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 加载保存的模型权重
    model.load_state_dict(torch.load(model_path))  # 修改为你实际的模型文件路径
    model.eval()  # 设置模型为评估模式

    # 演示动画
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # 渲染环境

        if isinstance(state, (list, tuple)) and len(state) != 8:
            print(f"Warning: Unexpected state length {len(state)}")
            state = np.zeros(8)

        state_tensor = torch.FloatTensor(np.array(state)).to(device).unsqueeze(0)
        action_probs = model(state_tensor)
        action = torch.argmax(action_probs).item()
        
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, done, truncated, _ = step_result
        elif len(step_result) == 4:
            next_state, reward, done, _ = step_result
        else:
            print(f"Warning: Unexpected step result length {len(step_result)}")
            next_state, reward, done, _ = (np.zeros(8), 0, True, {})

        total_reward += reward

        if isinstance(next_state, (list, tuple)) and len(next_state) != 8:
            print(f"Warning: Unexpected next_state length {len(next_state)}")
            next_state = np.zeros(8)

        state = next_state

    print(f"Total Reward: {total_reward}")
    
if __name__ == '__main__':
    num = 0
    model_path='D:\KAN-adversarial-attack-main\KAN_lunarlander\KAN_model_episode863.pth'
    env = gym.make('LunarLander-v2', render_mode="human")  # 使用 render_mode="human" 来渲染动画
    state_shape = env.observation_space.shape
    state_size = env.observation_space.shape[0]
    number_actions = env.action_space.n
    while True:
        show(env,state_size, number_actions,model_path)
        num+=1
        if num > 5:
            env.close()
            break
