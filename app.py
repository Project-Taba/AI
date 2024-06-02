from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import InputLayer
from collections import deque

# 기준값
thresholdMAE = 0.4252552301949038 * 0.9

app = Flask(__name__)

model = None    # AI 모델
max_accel, max_brake = 0, 0   # 엑셀, 브레이크 최대값
sensor_data = deque()  # 압력값을 저장할 큐
car_id = None   # 차량 id
driving_session_id = None  # 운전 세션 id
prev_speed = 0  # 이전 속도 값


# 모델 불러오기
def load_model_fn():
    global model
    model = load_model(
        './taba_model.h5',
        custom_objects={'mae': MeanAbsoluteError, 'InputLayer': InputLayer}
    )


@app.route("/calibration", methods=["POST"])
def calibration():
    global max_accel, max_brake, car_id  # 전역 변수로 사용

    data = request.get_json()
    sensorType = data.get('sensorType')
    car_id = data.get("carID")

    if sensorType == 'Accel':
        max_accel = data.get('pressureMax')
    elif sensorType == 'Brake':
        max_brake = data.get('pressureMax')
    else:
        return jsonify({"message": f"{car_id}의 Sensor Type이 정의되지 않았습니다."}), 400

    return jsonify({"message": "Calibration 성공"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    global prev_speed  # 전역 변수로 사용

    # 서버로부터 accel, brake, 현재 speed을 받음
    data = request.get_json()
    accel_value = data.get("accelPressure")
    brake_value = data.get("brakePressure")
    cur_speed = data.get("speed")

    shift_speed = cur_speed - prev_speed    # 속도 차이 저장
    prev_speed = cur_speed  # 이전 속도 저장

    # MinMaxScaling -> min값은 0으로 설정
    accel_value = accel_value / max_accel if max_accel else 0
    brake_value = brake_value / max_brake if max_brake else 0

    sensor_data.append([accel_value, brake_value, shift_speed])

    # 데이터가 30개 이상이면 모델 예측 수행
    if len(sensor_data) >= 30:
        # 모델 입력을 위해 np.array로 변환
        data_array = np.array([sensor_data])  # 모델 입력을 위해 차원을 맞춤
        Predict = model.predict(data_array)
        MAE = np.mean(np.abs(Predict - sensor_data[-1]), axis=1)

        if thresholdMAE < MAE:  # 이상치 발생
            result = "Error"
        else:
            result = "Normal"

        sensor_data.popleft()  # 가장 오래된 것 제거

        # 결과 return
        return jsonify({"message": f"{driving_session_id}의 결과", "result": result}), 202

    else:  # 데이터의 개수가 모자라다면
        return jsonify({"message": f"{driving_session_id}의 data 측정중"}), 202


if __name__ == "__main__":
    load_model_fn()    # 모델 불러오기
    app.run()       # 플라스크 실행
