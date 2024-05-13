# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import os
import seaborn as sns
from datetime import * 
from functools import reduce
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import streamlit as st

#Path불러오기 
folder_path = os.path.join("./output_data/")
daily_sema = pd.read_csv(f'{folder_path}'+'daily_fitbit_sema_df_unprocessed.csv')
hourly_sema = pd.read_csv(f'{folder_path}'+'hourly_fitbit_sema_df_unprocessed.csv')
breq = pd.read_csv(f'{folder_path}'+'breq.csv')
panas = pd.read_csv(f'{folder_path}'+'panas.csv')
personality = pd.read_csv(f'{folder_path}'+'personality.csv')
stai = pd.read_csv(f'{folder_path}'+'stai.csv')
ttm = pd.read_csv(f'{folder_path}'+'ttm.csv')

st.markdown(
    "<h1 style='text-align: center;'>Kaggle Fitbit Sleep EDA Project</h1><br><br>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h3>스트레스 관리능력과 운동의 관계</h3>"
    "<h4>stress_score와 exertion_points_percentage의 관계</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 stress_score와  exertion_points_percentage에 대한 회귀 그래프 입니다. 회귀 계수가 1.28임을 볼 때
stress_score와 exertion_points_percentage가 양의 상관 관계임을 확인할 수 있습니다.
스트레스 관리 능력이 높을 수록 운동 비율이 높아지는 경향을 보입니다.<br><br>
</div>
""", unsafe_allow_html=True)
def plot_stress_vs_exertion(daily_sema):
    plt.figure(figsize=(10, 6))
    daily_sema.dropna(subset=['exertion_points_percentage', 'stress_score'], inplace=True)
    subset_data = daily_sema[daily_sema['stress_score'] > 60]
    subset_data['exertion_points_percentage_scaled'] = subset_data['exertion_points_percentage'] * 100
    sns.regplot(x='stress_score', y='exertion_points_percentage_scaled', data=subset_data, line_kws={"color": "red"})
    plt.title('Stress Score vs. exertion_points_percentage (Stress Score > 60)')
    plt.xlabel('Stress Score')
    plt.ylabel('exertion_points_percentage')

    # 선형 회귀 모델 학습
    X = subset_data[['stress_score']]
    y = subset_data['exertion_points_percentage_scaled']
    model = LinearRegression()
    model.fit(X, y)

    # 회귀선 그리기
    plt.plot(X, model.predict(X), color='red', label=f'Regression Line (Coefficient: {model.coef_[0]:.2f})')

    plt.legend(loc='lower left')
    plt.tight_layout()  # 그래프 간 간격 조정
    st.pyplot(plt.gcf())  # 스트림릿에 그래프 출력

# daily_sema 데이터프레임을 가정하고 함수 호출
plot_stress_vs_exertion(daily_sema)
st.markdown(
    "<h4>stress_score와 total_active_minutes의 관계</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 stress_score와 total_active_minutes에 대한 그래프입니다. 회귀계수가 4.71일 정도 인 것을 볼 때
stress관리 능력이 높을 수록 움직인 시간이 강하게 증가하는 경향을 볼 수 있습니다.<br><br>
</div>
""", unsafe_allow_html=True)
def plot_stress_vs_activity(daily_sema):
    daily_sema['total_active_minutes'] = daily_sema['lightly_active_minutes'] + daily_sema['moderately_active_minutes'] + daily_sema['very_active_minutes']
    # NaN 값이 있는 행 제거
    daily_sema.dropna(subset=['total_active_minutes', 'stress_score'], inplace=True)

    # 독립 변수 선택
    X = daily_sema[daily_sema['stress_score'] > 60][['stress_score']]

    # 종속 변수 선택
    y = daily_sema[daily_sema['stress_score'] > 60]['total_active_minutes']

    # 시각화
    plt.figure(figsize=(8, 6))

    # Linear Regression 모델 생성
    model = LinearRegression()

    # 모델 피팅
    model.fit(X, y)

    # 회귀선 그리기
    sns.regplot(x='stress_score', y='total_active_minutes', data=daily_sema[daily_sema['stress_score'] > 60], line_kws={"color": "red"})
    # 회귀 계수 출력
    coeff = model.coef_[0]

    plt.title('stress_score vs. total_active_minutes (stress_score > 60)')
    plt.xlabel('stress_score')
    plt.ylabel('total_active_minutes')
    plt.plot(X, model.predict(X), color='red', label=f'Regression Line (Coefficient: {model.coef_[0]:.2f})')

    plt.legend(loc='lower left')
    plt.tight_layout()
    st.pyplot(plt.gcf())  # 스트림릿에 그래프 출력

# 함수 호출
plot_stress_vs_activity(daily_sema)
st.markdown(
    "<h4>stress_score와 activity_minutes type의 관계</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 stress_score와 각각의 active_minutes type과의 관계를 나타낸 그래프 입니다.
stress_score가 높을수록 very_active_minutes(1.01), moderate_active_minutes(1.14), light_active_minutes(2.56)이 높아지는 것을 볼 수 있습니다.
스트레스 관리 능력이 높아질 수록 활동시간은 늘어나지만 특히 약간 강도의 운동시간이 증가하는 경향을 보입니다.
반면 stess_score가 높을수록 sedentary_minutes(-3.73)의 관계는 감소하는 것을 확인 할 수 있습니다.
스트레스 관리 능력이 높아질 수록 휴식시간이 많이 줄어주고, 활동을 하는 경향을 보입니다. <br><br>
</div>
""", unsafe_allow_html=True)
def plot_activity_vs_stress_subplot(daily_sema, activity_type, ax):
    daily_sema.dropna(subset=[f'{activity_type}', 'stress_score'], inplace=True)
    X = daily_sema[daily_sema['stress_score'] > 60][['stress_score']]

    # 종속 변수 선택
    y = daily_sema[daily_sema['stress_score'] > 60][f'{activity_type}']

    # Linear Regression 모델 생성
    model = LinearRegression()

    # 모델 피팅
    model.fit(X, y)

    sns.regplot(x='stress_score', y=f'{activity_type}', data=daily_sema[daily_sema['stress_score'] > 60], ax=ax, line_kws={"color": "red"})

    # 회귀 계수 출력
    coeff = model.coef_[0]

    ax.set_title(f'stress_score vs. {activity_type} (stress_score > 60)')
    ax.set_xlabel('stress_score')
    ax.set_ylabel(f'{activity_type}')
    ax.plot(X, model.predict(X), color='red', label=f'Regression Line (Coefficient: {model.coef_[0]:.2f})')

    ax.legend(loc='lower left')
    plt.tight_layout()

# 2x2 서브플롯 생성
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 서브플롯에 그래프 그리기
plot_activity_vs_stress_subplot(daily_sema, 'very_active_minutes', axs[0, 0])
plot_activity_vs_stress_subplot(daily_sema, 'moderately_active_minutes', axs[0, 1])
plot_activity_vs_stress_subplot(daily_sema, 'lightly_active_minutes', axs[1, 0])
plot_activity_vs_stress_subplot(daily_sema, 'sedentary_minutes', axs[1, 1])

# 그래프 출력
st.pyplot(fig)  # 스트림릿에 그래프 출력

st.markdown(
    "<h3>스트레스 관리능력과 수면의 관계</h3>"
    "<h4>stress_score와 sleep_points_percentage의 관계</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 stress_score와 sleep_points_percentage의 관계를 나타낸 그래프입니다.
회귀계수가 0.93임을 보아 stress_score가 증가할 수록 sleep_points_percentage가 증가하는 것을 볼 수 있습니다.
즉 스트레스 관리 능력이 높아질 수록 수면의 질이 증가하는 경향을 보입니다.
<br><br>
</div>
""", unsafe_allow_html=True)
# 그래프 크기 지정
fig, ax = plt.subplots(figsize=(10, 6))

# stress_score가 60보다 큰 subset 데이터 생성
subset_data = daily_sema[daily_sema['stress_score'] > 60]
subset_data['sleep_points_scaled'] = subset_data['sleep_points_percentage'] * 100

# 회귀선 그리기
sns.regplot(x='stress_score', y='sleep_points_scaled', data=subset_data, line_kws={"color": "red"})

# 선형 회귀 모델 학습
X = subset_data[['stress_score']]
y = subset_data['sleep_points_scaled']
model = LinearRegression()
model.fit(X, y)

# 회귀선 그리기
ax.plot(X, model.predict(X), color='red', label=f'Regression Line (Coefficient: {model.coef_[0]:.2f})')
ax.legend(loc='lower left')
ax.set_xlabel('Stress Score')
ax.set_ylabel('sleep_points_percentage')
ax.set_title('Stress Score vs. sleep_points_percentage (Stress Score > 60)')

# 스트림릿에 그래프 출력
st.pyplot(fig)

st.markdown(
    "<h4>stress_score와 sleep_wake_ratio의 관계</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 stress_score와 sleep_wake_ratio의 관계를 나타낸 그래프입니다.
회귀계수가 -0.43임을 보아 stress_score가 증가할 수록 sleep_wake_ratio가 감소하는 것을 볼 수 있습니다.
즉 스트레스 관리 능력이 높아질 수록 잠에서 깨는 비율이 낮아짐을 확인할 수 있습니다.
<br><br>
</div>
""", unsafe_allow_html=True)
def plot_stress_vs_sleep_wake_ratio(daily_sema):
    daily_sema.dropna(subset=['sleep_wake_ratio', 'stress_score'], inplace=True)
    subset_data = daily_sema[daily_sema['stress_score'] > 60]
    subset_data['sleep_wake_ratio_scaled'] = subset_data['sleep_wake_ratio'] * 100

    # Linear Regression 모델 생성
    model = LinearRegression()

    # 독립 변수 선택
    X = subset_data[['stress_score']]

    # 종속 변수 선택
    y = subset_data['sleep_wake_ratio_scaled']

    # 모델 피팅
    model.fit(X, y)

    # 시각화
    plt.figure(figsize=(10, 6))

    # 회귀선 그리기
    sns.regplot(x='stress_score', y='sleep_wake_ratio_scaled', data=subset_data, line_kws={"color": "red"})
    plt.plot(X, model.predict(X), color='red', label=f'Regression Line (Coefficient: {model.coef_[0]:.2f})')

    plt.title('Stress Score vs. sleep_wake_ratio (Stress Score > 60)')
    plt.xlabel('Stress Score')
    plt.ylabel('sleep_wake_ratio')
    plt.legend(loc='lower left')
    plt.tight_layout()  # 그래프 간 간격 조정
    st.pyplot(plt.gcf())  # 스트림릿에 그래프 출력

# 함수 호출
plot_stress_vs_sleep_wake_ratio(daily_sema)

st.markdown(
    "<h4>exertion_points_percentage와 sleep_wake_ratio의 관계</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 exertion_points_percentage와 sleep_points_percentage의 관계를 나타낸 그래프입니다.
회귀계수가 거의 0에 가까운 것을 볼 때(-0.04) 관계가 없는 경향성을 보입니다.
상식과 다르게 운동비율과 수면의 질이 관계 없는 경향을 보입니다. 이 부분은 active_minutes를 통해 조금 더 확인할 필요가 있어 보입니다.
<br><br>
</div>
""", unsafe_allow_html=True)
def plot_exertion_vs_sleep_points(daily_sema):
    daily_sema.dropna(subset=['exertion_points_percentage', 'sleep_points_percentage'], inplace=True)

    # Subset data where exertion_points_percentage is available and stress_score > 60
    subset_data = daily_sema[daily_sema['stress_score'] > 60]

    # Linear Regression model
    model = LinearRegression()

    # Scaling sleep_points_percentage and exertion_points_percentage by 100
    subset_data['scaled_sleep_points'] = subset_data['sleep_points_percentage'] * 100
    subset_data['scaled_exertion_points'] = subset_data['exertion_points_percentage'] * 100

    # Independent variable
    X = subset_data[['scaled_exertion_points']]

    # Dependent variable
    y = subset_data['scaled_sleep_points']

    # Model fitting
    model.fit(X, y)

    # Visualization
    plt.figure(figsize=(10, 6))

    # Scatter plot with regression line
    sns.regplot(x='scaled_exertion_points', y='scaled_sleep_points', data=subset_data, line_kws={"color": "red"})
    plt.plot(X, model.predict(X), color='red', label=f'Regression Line (Coefficient: {model.coef_[0]:.2f})')

    plt.title('Exertion Points Percentage vs. Sleep Points Percentage (Stress Score > 60)')
    plt.xlabel('Scaled Exertion Points Percentage')
    plt.ylabel('Scaled Sleep Points Percentage')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(plt.gcf())

# Call the function
plot_exertion_vs_sleep_points(daily_sema)
