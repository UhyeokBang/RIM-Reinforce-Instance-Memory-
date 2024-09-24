import gym
import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gym import spaces

# 사용자 요청 가중치 계산 함수
def calculate_user_request_weight(total_requests, recent_requests, time_diff, decay_rate=0.1):
    time_decay_weight = math.exp(-decay_rate * time_diff)
    recent_request_ratio = recent_requests / total_requests if total_requests > 0 else 0
    weight = total_requests * time_decay_weight * recent_request_ratio
    return weight

# Custom Environment for Instance Memory Management with Dynamic Reward Adjustment
class InstanceMemoryEnv(gym.Env):
    def __init__(self, n_instances=5, scenario=None, total_memory_capacity=1000):
        super(InstanceMemoryEnv, self).__init__()
        self.n_instances = n_instances
        self.scenario = scenario
        self.current_step = 0

        # 총 메모리 용량 및 현재 사용량
        self.total_memory_capacity = total_memory_capacity
        self.current_memory_usage = 0  # 현재 사용 중인 InstanceView의 수

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
        self.request_history = [[] for _ in range(n_instances)]  # 각 인스턴스의 요청 시간 히스토리
        
        # 사용자 요청 가중치 초기화 (초기에는 모든 인스턴스에 대해 동일한 가중치)
        self.user_request_weight = np.random.rand(n_instances) * 5  # Request weight range: 0 to 5

        # Set initial view allocations
        self.current_views = np.random.randint(1, 5, size=n_instances)  # Initial views between 1 and 4
        self.max_views = 10  # Max views allowed for any instance
        self.update_memory_usage()  # 현재 메모리 사용량 업데이트

    def update_memory_usage(self):
        """
        현재 인스턴스의 총 메모리 사용량을 업데이트
        """
        self.current_memory_usage = np.sum(self.current_views)

    def reset(self):
        # Reset state at the beginning of each episode
        self.current_views = np.random.randint(1, 5, size=self.n_instances)
        self.current_step = 0
        self.state = np.stack((self.current_views, [self.max_views] * self.n_instances, 
                               self.instance_difficulty, self.user_request_weight), axis=1)
        self.update_memory_usage()
        return self.state

    def step(self, action):
        # Apply the actions: Increase/Decrease the view allocation
        reward = 0
        for i in range(self.n_instances):
            if action[i] == 0:  # Decrease views
                if self.current_views[i] > 1:
                    self.current_views[i] -= 1
                    reward += self.calculate_reward_for_decrease()  # 메모리 절약에 대한 보상
            elif action[i] == 2:  # Increase views
                if self.current_memory_usage < self.total_memory_capacity:  # 총 메모리 용량을 초과하지 않을 때만 증가 가능
                    if self.current_views[i] < self.max_views:
                        self.current_views[i] += 1
                        reward += self.calculate_reward_for_increase()  # 올바른 할당에 대한 보상
                else:
                    reward -= self.calculate_penalty_for_exceeding_memory()  # 메모리 초과 상태에서 증가하려는 경우 페널티

        # Update memory usage and state
        self.update_memory_usage()
        self.state[:, 0] = self.current_views

        # 시나리오에 따른 보상 계산 및 사용자 요청 가중치 업데이트
        if self.scenario and self.current_step < len(self.scenario):
            step, instance_id, user_request, success = self.scenario[self.current_step]
            
            if user_request == 1:  # 사용자가 요청한 경우
                self.total_requests[instance_id] += 1  # 총 요청 횟수 증가
                self.recent_requests[instance_id] += 1  # 최근 요청 횟수 증가
                time_diff = self.current_step - self.last_request_time[instance_id]  # 마지막 요청으로부터 경과한 시간
                self.last_request_time[instance_id] = self.current_step  # 마지막 요청 시점 업데이트

                # 시간에 따른 감쇠 적용
                self.recent_requests *= 0.9  # 최근 요청에 대해 일정 비율로 감소

                # 요청 시간 히스토리 업데이트
                self.request_history[instance_id].append(self.current_step)
                # 최근 10 스텝 동안의 요청 횟수로 업데이트
                self.recent_requests[instance_id] = len([step for step in self.request_history[instance_id] if self.current_step - step < 10])

                # 사용자 요청 가중치 업데이트
                self.user_request_weight[instance_id] = calculate_user_request_weight(
                    self.total_requests[instance_id], self.recent_requests[instance_id], time_diff
                )

                # 탐색 성공/실패에 따른 보상
                if success:
                    reward += 2
                else:
                    reward -= 1

        # 메모리 상태에 따른 추가 보상/페널티
        reward += self.dynamic_memory_reward_adjustment()

        # Determine if episode is done
        done = False
        if self.current_step >= len(self.scenario) - 1:
            done = True

        # Step 진행
        self.current_step += 1

        return self.state, reward, done, {}

    def calculate_reward_for_decrease(self):
        """
        메모리를 줄이는 행동에 대한 보상 계산
        """
        memory_threshold = 0.8 * self.total_memory_capacity  # 임계값
        if self.current_memory_usage > memory_threshold:
            return 1.0  # 메모리가 임계값에 가까울 때는 더 큰 보상
        return 0.5

    def calculate_reward_for_increase(self):
        """
        메모리를 올바르게 사용하는 행동에 대한 보상 계산
        """
        memory_threshold = 0.8 * self.total_memory_capacity
        if self.current_memory_usage < memory_threshold:
            return 1.0  # 메모리가 충분할 때는 더 큰 보상
        return 0.5

    def calculate_penalty_for_exceeding_memory(self):
        """
        메모리 초과 시 페널티 계산
        """
        return 3.0  # 메모리 초과 시 큰 페널티

    def dynamic_memory_reward_adjustment(self):
        """
        현재 메모리 사용량에 따라 보상 조정
        """
        reward_adjustment = 0
        memory_threshold = 0.8 * self.total_memory_capacity

        # 메모리가 충분할 때는 보상 증가
        if self.current_memory_usage < memory_threshold:
            reward_adjustment += 1

        # 메모리가 부족할 때는 페널티 증가
        elif self.current_memory_usage > memory_threshold:
            reward_adjustment -= 1

        # 메모리 초과 시 큰 페널티
        if self.current_memory_usage > self.total_memory_capacity:
            reward_adjustment -= 5

        return reward_adjustment

    def render(self, mode='human'):
        # Print the current state and allocation
        print(f"Current Step: {self.current_step}")
        print(f"Current Views: {self.current_views}")
        print(f"Current Memory Usage: {self.current_memory_usage}/{self.total_memory_capacity}")
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

# Create the environment with the scenario and memory constraints
env = DummyVecEnv([lambda: InstanceMemoryEnv(n_instances=5, scenario=scenario, total_memory_capacity=1000)])

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("instance_memory_model_with_dynamic_rewards_and_request_management")

# Load the model (for future use)
model = PPO.load("instance_memory_model_with_dynamic_rewards_and_request_management")

# Evaluate the model
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
