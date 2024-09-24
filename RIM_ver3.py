import gym
import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gym import spaces

# 사용자 요청 가중치 계산 함수
def calculate_user_request_weight(total_requests, recent_requests, time_diff, decay_rate=0.1):
    """
    사용자 요청 가중치 계산 함수

    Args:
        total_requests (int): 전체 요청 횟수
        recent_requests (int): 최근 일정 기간 동안의 요청 횟수
        time_diff (float): 마지막 요청으로부터 지난 시간
        decay_rate (float): 시간에 따른 가중치 감소율 (default: 0.1)

    Returns:
        weight (float): 사용자 요청 가중치
    """
    # 시간에 따른 가중치 감소 계산
    time_decay_weight = math.exp(-decay_rate * time_diff)

    # 최근 요청 비율 계산
    recent_request_ratio = recent_requests / total_requests if total_requests > 0 else 0

    # 사용자 요청 가중치 계산
    weight = total_requests * time_decay_weight * recent_request_ratio
    return weight

# Custom Environment for Instance Memory Management with Scenarios
class InstanceMemoryEnv(gym.Env):
    def __init__(self, n_instances=5, scenario=None):
        super(InstanceMemoryEnv, self).__init__()
        self.n_instances = n_instances
        self.scenario = scenario
        self.current_step = 0
        
        # Define the action and observation space
        self.action_space = spaces.MultiDiscrete([3] * n_instances)  # 0: Decrease, 1: No change, 2: Increase
        self.observation_space = spaces.Box(low=0, high=10, shape=(n_instances, 4), dtype=np.float32)
        
        # Initialize state
        self.state = np.zeros((n_instances, 4))
        
        # Set instance difficulty and user request weight (for simplicity, fixed values)
        self.instance_difficulty = np.random.rand(n_instances) * 5  # Difficulty range: 0 to 5

        # 사용자 요청 관련 정보 초기화
        self.total_requests = np.zeros(n_instances)  # 각 인스턴스의 총 요청 횟수
        self.recent_requests = np.zeros(n_instances)  # 최근 요청 횟수
        self.last_request_time = np.zeros(n_instances)  # 마지막 요청 시점 (스텝 기준)
        
        # 사용자 요청 가중치 초기화 (초기에는 모든 인스턴스에 대해 동일한 가중치)
        self.user_request_weight = np.random.rand(n_instances) * 5  # Request weight range: 0 to 5

        # Set initial view allocations
        self.current_views = np.random.randint(1, 5, size=n_instances)  # Initial views between 1 and 4
        self.max_views = 10  # Max views allowed for any instance

    def reset(self):
        # Reset state at the beginning of each episode
        self.current_views = np.random.randint(1, 5, size=self.n_instances)
        self.current_step = 0
        self.state = np.stack((self.current_views, [self.max_views] * self.n_instances, 
                               self.instance_difficulty, self.user_request_weight), axis=1)
        return self.state

    def step(self, action):
        # Apply the actions: Increase/Decrease the view allocation
        reward = 0
        for i in range(self.n_instances):
            if action[i] == 0:  # Decrease views
                self.current_views[i] = max(1, self.current_views[i] - 1)
            elif action[i] == 2:  # Increase views
                self.current_views[i] = min(self.max_views, self.current_views[i] + 1)
        
        # Update state
        self.state[:, 0] = self.current_views

        # 시나리오에 따른 보상 계산 및 사용자 요청 가중치 업데이트
        if self.scenario and self.current_step < len(self.scenario):
            step, instance_id, user_request, success = self.scenario[self.current_step]
            
            if user_request == 1:  # 사용자가 요청한 경우
                self.total_requests[instance_id] += 1  # 총 요청 횟수 증가
                self.recent_requests[instance_id] += 1  # 최근 요청 횟수 증가
                time_diff = self.current_step - self.last_request_time[instance_id]  # 마지막 요청으로부터 경과한 시간
                self.last_request_time[instance_id] = self.current_step  # 마지막 요청 시점 업데이트
                
                # 사용자 요청 가중치 업데이트
                self.user_request_weight[instance_id] = calculate_user_request_weight(
                    self.total_requests[instance_id], self.recent_requests[instance_id], time_diff
                )

                # 탐색 성공/실패에 따른 보상
                if success:
                    reward += 2
                    self.user_request_weight[instance_id] += 0.5  # 성공 시 가중치 증가
                else:
                    reward -= 1
                    self.user_request_weight[instance_id] += 1.0  # 실패 시 가중치 크게 증가

        # Determine if episode is done
        done = False
        if self.current_step >= len(self.scenario) - 1:
            done = True

        # Step 진행
        self.current_step += 1

        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Print the current state and allocation
        print(f"Current Step: {self.current_step}")
        print(f"Current Views: {self.current_views}")
        print(f"State: {self.state}")
        print(f"User Request Weights: {self.user_request_weight}")


# 시나리오 정의
scenario = [
    (0, 3, 1, True),  # Step 0: instance 3 탐색 요청, 성공
    (1, 1, 1, False), # Step 1: instance 1 탐색 요청, 실패
    (2, 4, 1, True),  # Step 2: instance 4 탐색 요청, 성공
    (3, 2, 1, False), # Step 3: instance 2 탐색 요청, 실패
    (4, 0, 1, True),  # Step 4: instance 0 탐색 요청, 성공
]

# Create the environment with the scenario
env = DummyVecEnv([lambda: InstanceMemoryEnv(n_instances=5, scenario=scenario)])

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("instance_memory_model_with_scenario")

# Load the model (for future use)
model = PPO.load("instance_memory_model_with_scenario")

# Evaluate the model
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
