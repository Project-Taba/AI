import numpy as np
import keras
from keras.metrics import MeanAbsoluteError
from collections import deque

if __name__ == "__main__":

    max_accel, max_brake = 764, 975
    predict_speed = None

    # 사용자 정의 메트릭을 custom_objects로 제공
    model = keras.models.load_model(
        './AI_two_input.h5',
        custom_objects={'mae': MeanAbsoluteError()}
    )

    # DRIVING, NONE -> string
    drive_complete = False
    prev_speed = 0
    shift_speed = 0

    data = deque()

    while True:
        try:
            if drive_complete:
                print("운전 완료, 시스템 종료")
                break

            # 서버로부터 accel, brake, 현재 speed을 받음
            # accel_value = 987654321
            # brake_value = 987654321
            # cur_speed = 987654321
            brake_value, accel_value, cur_speed = map(int, input().split())

            shift_speed = cur_speed - prev_speed    # 속도 차이 저장
            prev_speed = cur_speed  # 이전 속도 저장

            # MinMaxScaling -> min값은 0으로 설정
            accel_value /= max_accel
            brake_value /= max_brake

            data.append([accel_value, brake_value])

            # 데이터가 n개 이상이면 모델 예측 수행
            if len(data) >= 5:
                data_array = np.array([data])  # 모델 입력을 위해 차원을 맞춤
                predict_speed = model.predict(data_array)

                # # 모델 입력을 위해 np.array로 변환
                # if predict_speed != None:
                print("predict value: ", predict_speed)
                print("shift speed: ", shift_speed)
                if abs(predict_speed - shift_speed) > 5:
                    print("ERROR")
                else:
                    print("NORMAL")

                    # if thresholdMAE < Pred_MAE:  # 이상치 발생
                    #     result = "Error"
                    # else:
                    #     result = "Normal"

                data.popleft()  # 가장 오래된 것 제거
        except:
            print("oops~!")
