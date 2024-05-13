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

# Path불러오기 
folder_path = os.path.join("./output_data/")
daily_sema = pd.read_csv(f'{folder_path}'+'daily_fitbit_sema_df_unprocessed.csv')
hourly_sema = pd.read_csv(f'{folder_path}'+'hourly_fitbit_sema_df_unprocessed.csv')
breq = pd.read_csv(f'{folder_path}'+'breq.csv')
panas = pd.read_csv(f'{folder_path}'+'panas.csv')
personality = pd.read_csv(f'{folder_path}'+'personality.csv')
stai = pd.read_csv(f'{folder_path}'+'stai.csv')
ttm = pd.read_csv(f'{folder_path}'+'ttm.csv')

# daily_sema 데이터로 데이터프레임 생성
df = pd.DataFrame(daily_sema)

# 결측값 제거
df.dropna(subset=['sleep_duration'], inplace=True)

# 밀리초를 시간으로 변환하는 함수
def milliseconds_to_hours(milliseconds):
    hours = milliseconds / (1000 * 60 * 60)
    return hours

# 'sleep_duration' 컬럼의 값을 변환하여 새로운 컬럼에 저장
df['sleep_duration_hours'] = df['sleep_duration'].apply(milliseconds_to_hours)

# 'sleep_duration_hours' 컬럼 출력
print(df['sleep_duration_hours'])

# sleep_points_percentage 값을 100배하여 퍼센트로 표시
df['sleep_points_percentage'] = df['sleep_points_percentage'] * 100

# ID 별로 그룹화하여 평균값 계산
grouped_mean = df.groupby('id').mean()

st.markdown(
    "<h1 style='text-align: center;'>수면 효율성 및 수면 포인트 분석</h1>", 
    unsafe_allow_html=True
)
st.markdown(    
    "<h2>수면 평균 시간 및 수면 포인트 비율 관계 분석</h2>", 
    unsafe_allow_html=True
)

# Sleep Points Percentage vs Sleep Duration
plt.figure(figsize=(8, 6))
sns.regplot(data=grouped_mean, x='sleep_duration_hours', y='sleep_points_percentage')
plt.xlim(5, 10)
plt.title('Sleep Duration vs Sleep Points Percentage')
plt.xlabel('Sleep Duration (hours)')
plt.ylabel('Sleep Points Percentage')
st.pyplot(plt)

st.markdown("""**수면 시간과 수면 포인트 비율 비교**    
            → 수면 시간은 **6~8시간**으로 분포 되어있으며  
            → 수면 포인트 비율성은 **30~80퍼센트**로 고르게 분포되어 있음    
            → 전체적으로 봤을때 **양의 상관성**을 띈다 BUT 하지만 분포도로 따졌을때 **6~8시간** 수면 시간이 제일 **높은 비율**이 나옴""")

st.markdown(
    "<h2>수면 효율성 및 수면 포인트 비율 관계 분석</h2>", 
    unsafe_allow_html=True
)

# Sleep Points Percentage vs Sleep Effficiency
plt.figure(figsize=(8, 6))
sns.regplot(data=grouped_mean, x='sleep_efficiency', y='sleep_points_percentage')
plt.xlim(90, 98)
plt.title('Sleep Points Percentage vs Sleep Effficiency')
plt.xlabel('Sleep Efficiency')
plt.ylabel('Sleep Points Percentage')
st.pyplot(plt)

st.markdown("""**수면 효율과 수면 포인트 비율 비교**  
            → 수면 효율은 **90~98퍼센트** 사이에 고르게 분포 됨  
            → 수면 포인트 비율은 **20~80퍼센트**로 고르게 분포 됨  
            → 대체적으로 **양의 상관성**을 띔  
            → 수면 효율이 **높을수록** 수면 포인트 비율이 **높음**""")

#total_active_minutes 생성
daily_sema['total_active_minutes'] = daily_sema['lightly_active_minutes'] + daily_sema['moderately_active_minutes'] + daily_sema['very_active_minutes']

# 소수점 2자리 이하 반올림
daily_sema = daily_sema.round(2)

# sleep_points_percentage, sleep_efficiency이 NaN인 행 삭제
daily_sema = daily_sema.dropna(subset=['sleep_points_percentage', 'sleep_efficiency'])

# sleep_points_percentage, stress_score, total_active_minutes가 0인 경우 해당 행 삭제
daily_sema.drop(daily_sema[daily_sema['sleep_points_percentage'] == 0.0].index, inplace=True)
daily_sema.drop(daily_sema[daily_sema['stress_score'] == 0.0].index, inplace=True)
daily_sema.drop(daily_sema[daily_sema['total_active_minutes'] == 0.0].index, inplace=True)

st.markdown(
    "<h1 style='text-align: center;'>Exertion_points_percentage(활동지수)는 무엇인가?</h1>",
)

st.markdown(
    "<h2 style='text-align: center;'>Exertion_points_percentage(활동지수)에 영향을 미치는 요인</h2>"
    "<h3 style='text-align: center;'>Heatmap Top 10</h3>", 
    unsafe_allow_html=True
)

exertion_selcted_columns = ['exertion_points_percentage', 'stress_score', 'calories', 'distance', 'lightly_active_minutes', 'moderately_active_minutes', 'very_active_minutes', 'total_active_minutes', 'sedentary_minutes', 'resting_hr']
exertion_corr = daily_sema[exertion_selcted_columns].corr()

plt.figure()
sns.heatmap(data=exertion_corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
st.pyplot(plt)

st.markdown(
    "<h3 style='text-align: center;'>exertion_points_percentage에 영향을 미치는 요인 8개</h3>", 
    unsafe_allow_html=True
)
# 그래프 크기 및 subplot 생성
fig, axs = plt.subplots(3, 3, figsize=(18, 14))
axs[-1, -1].remove()  # 마지막 subplot 삭제

# 사용할 x축 값들
x_columns = ['calories', 'distance', 'total_active_minutes', 
             'lightly_active_minutes', 'moderately_active_minutes', 
             'very_active_minutes', 'sedentary_minutes', 'resting_hr']

# 각 subplot에 대해 line plot 그리기
for i, ax in enumerate(axs.flat):
    # x축과 y축 설정
    if i < len(x_columns):
        y_column = x_columns[i]
        x_column = 'exertion_points_percentage'

        # line plot 그리기
        sns.lineplot(x=x_column, y=y_column, data=daily_sema, ax=ax)

        # x축과 y축 레이블 설정
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

# 레이아웃 조정
plt.tight_layout()

# 그래프 출력
plt.show()

st.markdown(
    "<h3 style='text-align: center;'>calories, distance, total_active_minutes 처럼 활동과 관련된 지표가 상승할수록 exertion_points_percentage는 상승하며, \
        반대로 sedentary_minutes와 resting_hr처럼 휴식 지표가 상승할수록 exertion_points_percentage가 하락한다. \
            따라서 exertion_points_percentage가 활동지수로서 신뢰할 만한 지표인 것을 확인할 수 있다.</h3>",
    unsafe_allow_html=True
)

st.pyplot(plt)