import numpy as np
import keras
from keras.metrics import MeanAbsoluteError
from collections import deque

from calibration import Calibration

thresholdMAE = 0.4252552301949038 * 0.9

if __name__ == "__main__":
    calibration = Calibration()  # 객체 생성

    max_accel, max_brake = calibration.get_value()

    # 사용자 정의 메트릭을 custom_objects로 제공
    model = keras.models.load_model(
        './taba_model.h5',
        custom_objects={'mae': MeanAbsoluteError()}
    )

    # DRIVING, NONE -> string
    drive_complete = False
    prev_speed = 0
    shift_speed = 0

    data = deque()

    while True:
        if drive_complete:
            print("운전 완료, 시스템 종료")
            break

        # 서버로부터 accel, brake, 현재 speed을 받음
        accel_value = 987654321
        brake_value = 987654321
        cur_speed = 987654321

        shift_speed = cur_speed - prev_speed    # 속도 차이 저장
        prev_speed = cur_speed  # 이전 속도 저장

        # MinMaxScaling -> min값은 0으로 설정
        accel_value /= max_accel
        brake_value /= max_brake

        data.append([accel_value, brake_value, shift_speed])

        # 데이터가 30개 이상이면 모델 예측 수행
        if len(data) >= 30:
            # 모델 입력을 위해 np.array로 변환
            data_array = np.array([data])  # 모델 입력을 위해 차원을 맞춤
            Predict = model.predict(data_array)
            MAE = np.mean(np.abs(Predict - data[-1]), axis=1)

            if thresholdMAE < MAE:  # 이상치 발생
                result = "Error"
            else:
                result = "Normal"

            data.popleft()  # 가장 오래된 것 제거
