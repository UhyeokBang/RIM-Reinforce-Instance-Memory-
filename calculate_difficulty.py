import torch
import numpy as np

# 예시 포인트 클라우드 데이터와 객체 바운딩 박스 데이터 정의
# 실제 데이터에서는 Instance 클래스의 point_cloud 및 bounds 속성을 사용합니다.

# 객체의 바운딩 박스 (3, 2) 텐서: (x_min, x_max), (y_min, y_max), (z_min, z_max)
# 예시: (3 x 2) 텐서로 객체의 바운딩 박스 정보를 나타냅니다.
example_bounds = torch.tensor([[0, 1], [0, 1], [0, 0.5]])

# 객체의 포인트 클라우드 (N x 3) 텐서: N개의 (x, y, z) 좌표
# 예시: 100개의 랜덤 포인트 생성
example_point_cloud = torch.rand((100, 3))

def calculate_size_difficulty(bounds):
    """
    객체의 크기를 기반으로 난이도를 계산합니다.
    
    Args:
        bounds (Tensor): (3, 2) 크기의 텐서, 객체의 바운딩 박스 정보.

    Returns:
        size_difficulty (float): 크기 기반 난이도.
    """
    # 바운딩 박스 크기 계산
    size = (bounds[:, 1] - bounds[:, 0]).prod().item()  # (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    
    # 크기가 작을수록 난이도가 높음
    size_difficulty = 1 / size if size > 0 else float('inf')
    
    return size_difficulty

def calculate_shape_complexity(point_cloud):
    
    # 별로 현실적이지 않음

    """
    객체의 포인트 클라우드를 기반으로 형태 복잡성을 계산합니다.
    
    Args:
        point_cloud (Tensor): (N, 3) 크기의 텐서, 객체의 포인트 클라우드 정보.

    Returns:
        shape_complexity (float): 형태 복잡성 지수.
    """
    # 각 포인트의 곡률을 계산 (예: 각 포인트의 거리 평균값)
    mean_curvature = torch.mean(torch.norm(point_cloud - point_cloud.mean(dim=0), dim=1)).item()
    
    # 복잡성 지수는 곡률의 평균 값으로 설정 (곡률이 클수록 복잡성이 높아짐)
    shape_complexity = mean_curvature
    
    # 꼭지점 개수와 곡률에 따라 복잡성 지수 설정
    # 추가적으로 edge 개수, 볼록/오목성 등을 반영할 수 있음
    return shape_complexity

def calculate_instance_difficulty(instance):
    """
    객체의 크기 및 형태 복잡성을 바탕으로 난이도를 계산합니다.
    
    Args:
        instance (Instance): Instance 객체.

    Returns:
        difficulty (float): 최종 난이도.
    """
    # 크기 기반 난이도 계산
    size_difficulty = calculate_size_difficulty(instance.bounds)
    
    # 형태 복잡성 계산
    shape_complexity = calculate_shape_complexity(instance.point_cloud)
    
    # 최종 난이도 계산 (복잡성 지수를 곱하여 조절)
    difficulty = size_difficulty + shape_complexity * 0.5  # 복잡성에 상수 계수 곱함
    
    return difficulty

# 예제 인스턴스 정의 (실제 데이터에서는 Instance 클래스를 사용)
class Instance:
    bounds: torch.Tensor
    point_cloud: torch.Tensor

# 예제 객체 인스턴스 생성
example_instance = Instance(bounds=example_bounds, point_cloud=example_point_cloud)

# 난이도 계산
instance_difficulty = calculate_instance_difficulty(example_instance)
print(f"객체 난이도: {instance_difficulty:.4f}")
