import math

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
