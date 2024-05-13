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

import streamlit as st
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="시간대별 운동량과 수면의 질의 관계")
st.markdown("운동량은 칼로리 소모량으로 간접적으로 추론함")
# ###모든 글자는 st.markdown(
#     "<h2 style='text-align: center;'>스트레스 관리와 수면의 질, 활동과의 상관관계</h2>"
#     "<h3 style='text-align: center;'>Top10 heatmap</h3>",
#     unsafe_allow_html=True
# )###로 표현했음

#Path불러오기
folder_path = os.path.join("./output_data/")
daily_sema = pd.read_csv(f'{folder_path}'+'daily_fitbit_sema_df_unprocessed.csv')
hourly_sema = pd.read_csv(f'{folder_path}'+'hourly_fitbit_sema_df_unprocessed.csv')
breq = pd.read_csv(f'{folder_path}'+'breq.csv')
panas = pd.read_csv(f'{folder_path}'+'panas.csv')
personality = pd.read_csv(f'{folder_path}'+'personality.csv')
stai = pd.read_csv(f'{folder_path}'+'stai.csv')
ttm = pd.read_csv(f'{folder_path}'+'ttm.csv')


## 데이터 전처리 ##
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
    "<h1 style='text-align: center;'>평균 칼로리 소모량 상위 25%인 그룹과 하위 25%인 그룹을 추출하여\
        시간대별 칼로리 소모량을 추적함</h1>",
    unsafe_allow_html=True
)

# 'id' 별 'calories'의 평균 계산
calories_mean_by_id = daily_sema.groupby('id')['calories'].mean()

# 상위 25%와 하위 25%를 나누는 기준값 계산
quantile_75 = calories_mean_by_id.quantile(0.75)
quantile_25 = calories_mean_by_id.quantile(0.25)

# 상위 25%와 하위 25%에 해당하는 id 리스트 생성
upper_25_ids = calories_mean_by_id[calories_mean_by_id > quantile_75].index.tolist()
lower_25_ids = calories_mean_by_id[calories_mean_by_id < quantile_25].index.tolist()

print("칼로리 소모 상위 25%의 id 리스트:", upper_25_ids)
print("칼로리 소모 하위 25%의 id 리스트:", lower_25_ids)


st.markdown(
    "<h2 style='text-align: center;'>평균 칼로리 소모량 상위 25%인 그룹의 시간대별 칼로리 소모량 곡선\
        </h2>",
    unsafe_allow_html=True
)

upper_25_ids

i = 1
for id in upper_25_ids:
    plt.figure(figsize=(30, 20))
    plt.subplot(5, 5, i)
    sns.lineplot(data=hourly_sema.loc[hourly_sema['id'] == id], x='hour', y='calories')
    plt.title(id)
    i += 1
    st.pyplot(plt)

st.markdown(
    "<h2 style='text-align: center;'>평균 칼로리 소모량 하위 25%인 그룹의 시간대별 칼로리 소모량 곡선\
        </h2>",
    unsafe_allow_html=True
)

lower_25_ids

i = 1
for id in lower_25_ids:
    plt.figure(figsize=(30, 20))
    plt.subplot(5, 5, i)
    sns.lineplot(data=hourly_sema.loc[hourly_sema['id'] == id], x='hour', y='calories')
    plt.title(id)
    i += 1
    st.pyplot(plt)

st.markdown(
    "<h3 style='text-align: center;'>상위 25%인 그룹의 저녁 시간대(7~9시 사이) 운동량이 하위 25%에 비해 월등이 높다.\
        </h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center;'>상위 25%인 그룹(파란색)과 하위 25%인 그룹(오렌지색)의 시간대별 운동량.\
        </h2>",
    unsafe_allow_html=True
)

combined_data_upper = pd.concat([hourly_sema.loc[hourly_sema['id'] == id] for id in upper_25_ids])
combined_data_lower = pd.concat([hourly_sema.loc[hourly_sema['id'] == id] for id in lower_25_ids])

# kdeplot으로 데이터의 분포 시각화
plt.figure(figsize=(10, 10))
sns.kdeplot(data=combined_data_upper, x='hour', y='calories', color='blue', label='upper 25%', shade=True)

sns.kdeplot(data=combined_data_lower, x='hour', y='calories', color='orange', label='lower 25%', shade=True)
plt.ylim(0, 400)
plt.title('Combined KDE Plot for lower 25% ids vs upper 25% ids')
plt.legend(loc='upper left', fontsize='large')
st.pyplot(plt)

st.markdown(
    "<h2 style='text-align: center;'>상위 25%인 그룹(파란색)은 마치 낙타처럼 저녁 시간대의 운동량이 급격하게 상승한 모습을 볼 수 있다.\
        </h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center;'>즉 저녁 시간대의 운동은 수면 품질에 악영향을 미칠 수 있다.\
        </h1>",
    unsafe_allow_html=True
)