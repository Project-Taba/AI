import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.metrics import MeanAbsoluteError
import seaborn as sns

trainMAE = 0.4252552301949038

# 시퀀스 데이터 생성 함수


def to_sequences(x, seq_size=1):
    x_values = []
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
    return np.array(x_values)


# 사용자 정의 메트릭을 custom_objects로 제공
loaded_model = keras.models.load_model(
    './taba_model.h5',
    custom_objects={'mae': MeanAbsoluteError()}
)

print(loaded_model.summary())
