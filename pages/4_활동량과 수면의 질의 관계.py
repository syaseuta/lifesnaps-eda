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

st.set_page_config(page_title="활동량(Active_minutes)와 수면의 질(sleep_points_percentage)의 관계")
st.markdown("활동량(Active_minutes)와 수면의 질(sleep_points_percentage)의 관계")
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
    "<h1 style='text-align: center;'>활동 지수에 큰 영향을 미치는 요소를 종류별로 나누어\
        sleep_points_percetnage(수면품질)과의 연관성 분석</h1>",
    unsafe_allow_html=True
)

plt.figure(figsize=(20, 20))
plt.subplot(3, 2, 1)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='calories', palette='bright')
plt.title('calories vs sleep_points_percentage')
plt.subplot(3, 2, 2)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='distance', palette='bright')
plt.title('distance vs sleep_points_percentage')
plt.subplot(3, 2, 3)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='total_active_minutes', palette='bright')
plt.title('total_active_minutes vs sleep_points_percentage')
plt.subplot(3, 2, 4)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='lightly_active_minutes', palette='bright')
plt.title('lightly_active_minutes vs sleep_points_percentage')
plt.subplot(3, 2, 5)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='moderately_active_minutes', palette='bright')
plt.title('moderately_active_minutes vs sleep_points_percentage')
plt.subplot(3, 2, 6)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='very_active_minutes', palette='bright')
plt.title('very_active_minutes vs sleep_points_percentage')
st.pyplot(plt)

st.markdown(
    "<h2 style='text-align: center;'>Active_minutes가 증가할수록\
        sleep_points_percetnage(수면품질)이 하락하는 경향을 보임.</h2>",
    unsafe_allow_html=True
)

plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='sedentary_minutes', palette='bright')
plt.title('sedentary_minutes vs sleep_points_percentage')
plt.subplot(1, 2, 2)
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='resting_hr', palette='bright')
plt.title('resting_hr vs sleep_points_percentage')
st.pyplot(plt)

st.markdown(
    "<h2 style='text-align: center;'>휴식 시간(sedentary_minutes, resting_hr)이 증가할수록\
        sleep_points_percetnage(수면품질)이 상승하는 경향을 보임.</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>운동시간(active_minutes)가 증가할수록 \
        sleep_points_percetnage(수면품질)이 하락하는 이유가 무엇일까?</h3>",
    unsafe_allow_html=True
)