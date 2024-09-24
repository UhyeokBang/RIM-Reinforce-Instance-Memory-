import matplotlib.pyplot as plt
import numpy as np
import cv2

# 인스턴스 메모리에서 인스턴스를 찾는 함수 (bounds 사용)
def find_instance_at(x, y, instance_memory):
    # 각 인스턴스의 3D bounds를 확인하여 해당 좌표가 포함되는지 검사
    for instance in instance_memory.instance_views:
        if instance.bounds is not None:
            # bounds는 3x2 크기의 텐서로, 최소값과 최대값을 나타냄
            x_min, y_min, z_min = instance.bounds[:, 0]
            x_max, y_max, z_max = instance.bounds[:, 1]

            # 좌표가 인스턴스의 3D bounds 내에 있는지 확인
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return instance
    return None


# 마우스 이벤트 처리 함수
def on_mouse_move(event, instance_memory):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        instance = find_instance_at(x, y, instance_memory)
        if instance:
            # 최적의 뷰에서 이미지를 가져옴
            best_view = instance.get_best_view()
            if best_view and best_view.cropped_image is not None:
                # 인스턴스 이미지 출력
                cv2.imshow(f"Instance at {x}, {y}", best_view.cropped_image)
                # cv2.waitKey(1000)  # 1초 동안 이미지 표시 후 닫기
                # cv2.destroyAllWindows()

# 맵 시각화
fig, ax = plt.subplots()
semantic_map = np.random.random((100, 100))  # 예시 맵
ax.imshow(semantic_map, cmap='gray')

instance_memory = {}

# 마우스 이벤트 리스너 연결
fig.canvas.mpl_connect('motion_notify_event', lambda event: on_mouse_move(event, instance_memory))

# 마우스 클릭으로 하는 경우
# fig.canvas.mpl_connect('button_press_event', lambda event: on_mouse_click(event, instance_memory))

plt.show()
