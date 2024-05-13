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
st.markdown(
    "<h1 style='text-align: center;'>Kaggle Fitbit Sleep EDA Project</h1><br><br>", 
    unsafe_allow_html=True
)
# ###모든 글자는 st.markdown(
#     "<h2 style='text-align: center;'>스트레스 관리와 수면의 질, 활동과의 상관관계</h2>"
#     "<h3 style='text-align: center;'>Top10 heatmap</h3>",
#     unsafe_allow_html=True
# )###로 표현했음

#Path불러오기
folder_path = os.path.join("../output_data/")
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

# daily_sema 데이터프레임에서 'sedentary_minutes'의 75번째 백분위수 계산
sedentary_quantile_75 = daily_sema['sedentary_minutes'].quantile(0.75)

# 'total_active_minutes'가 75번째 백분위수 이상인지를 확인하고 'total_active_rank' 설정
daily_sema['sedentary_rank'] = daily_sema['sedentary_minutes'].apply(
    lambda x: 'upper 25%' if x >= sedentary_quantile_75 else 'upper 75%'
)

# daily_sema 데이터프레임에서 'total_active_minutes'의 75번째 백분위수 계산
total_active_quantile_75 = daily_sema['total_active_minutes'].quantile(0.75)

# 'total_active_minutes'가 75번째 백분위수 이상인지를 확인하고 'total_active_rank' 설정
daily_sema['total_active_rank'] = daily_sema['total_active_minutes'].apply(
    lambda x: 'upper 25%' if x >= total_active_quantile_75 else 'upper 75%'
)

# daily_sema 데이터프레임에서 'lightly_active_minutes'의 75번째 백분위수 계산
lightly_quantile_75 = daily_sema['lightly_active_minutes'].quantile(0.75)

# 'lightly_active_minutes'가 75번째 백분위수 이상인지를 확인하고 'lightly_active_rank' 설정
daily_sema['lightly_active_rank'] = daily_sema['lightly_active_minutes'].apply(
    lambda x: 'upper 25%' if x >= lightly_quantile_75 else 'upper 75%'
)

# daily_sema 데이터프레임에서 'moderately_active_minutes'의 75번째 백분위수 계산
moderately_quantile_75 = daily_sema['moderately_active_minutes'].quantile(0.75)

# 'moderately_active_minutes'가 75번째 백분위수 이상인지를 확인하고 'moderately_rank' 설정
daily_sema['moderately_active_rank'] = daily_sema['moderately_active_minutes'].apply(
    lambda x: 'upper 25%' if x >= moderately_quantile_75 else 'upper 75%'
)

# daily_sema 데이터프레임에서 'very_active_minutes'의 75번째 백분위수 계산
very_active_quantile_75 = daily_sema['very_active_minutes'].quantile(0.75)

# 'very_active_minutes'가 75번째 백분위수 이상인지를 확인하고 'very_active_rank' 설정
daily_sema['very_active_rank'] = daily_sema['very_active_minutes'].apply(
    lambda x: 'upper 25%' if x >= very_active_quantile_75 else 'upper 75%'
)



st.markdown(
    "<h3 style='text-align: left;'>다음 그래프는 활동 시간을 강도별로 나누어 수면품질과 스트레스 관리 수치의 연관성을 분석한 결과입니다.</h3>"
    "<h4 style='text-align: left;'>stress_score vs sleep_points_percentage by 'active_minutes'.</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 총 운동 시간, 가벼운 운동 시간, 적당한 운동 시간, 강한 운동 시간에 따라 수면 품질이 어떻게 달라지는지 분석한 그래프입니다. 
운동의 강도별로 운동량 상위 25%인 그룹과 상위 75%인 그룹을 구분하여 수면품질의 차이를 분석한 결과
스트레스 관리 능력이 높을 수록 수면 품질이 높아지는 경향을 보입니다.
주목할 점은 운동량 상위 75%인 그룹이 운동량 상위 25%인 그룹에 비해 높은 수면품질을 보여준다는 것입니다.<br><br>
</div>
""", unsafe_allow_html=True)


plt.figure(figsize=(20, 20))
plt.subplot(3, 2, 1)
sns.lineplot(data=daily_sema, x='stress_score', y='sleep_points_percentage', hue='sedentary_rank', palette='bright')
plt.title('sedentary_rank')
plt.subplot(3, 2, 2)
sns.lineplot(data=daily_sema, x='stress_score', y='sleep_points_percentage', hue='total_active_rank', palette='bright')
plt.title('Total_Active_rank')
plt.subplot(3, 2, 3)
sns.lineplot(data=daily_sema, x='stress_score', y='sleep_points_percentage', hue='lightly_active_rank', palette='bright')
plt.title('lightly_active_rank')
plt.subplot(3, 2, 4)
sns.lineplot(data=daily_sema, x='stress_score', y='sleep_points_percentage', hue='moderately_active_rank', palette='bright')
plt.title('moderately_active_rank')
plt.subplot(3, 2, 5)
sns.lineplot(data=daily_sema, x='stress_score', y='sleep_points_percentage', hue='very_active_rank', palette='bright')
plt.title('very_active_rank')
plt.show()
st.pyplot(plt)

st.markdown(
    "<h4 style='text-align: left;'>즉 상식과 달리 Active_minutes가 높으면 수면품질이 오히려 떨어지는 결과가 나타난다는 것입니다.</h4>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: left;'>다음 그래프는 휴식 시간과 수면 품질의 관련성을 나타낸 그래프입니다.</h3>"
    "<h4 style='text-align: left;'>resting_hr vs sleep_points_percentage</h4>",
    unsafe_allow_html=True
)

plt.figure(figsize=(20, 20))
sns.lineplot(data=daily_sema, x='sleep_points_percentage', y='resting_hr', palette='bright')
plt.title('resting_hr vs sleep_points_percentage')
st.pyplot(plt)

st.markdown(
    "<h4 style='text-align: left;'>오히려 휴식 시간(resting_hr)이 증가할수록 sleep_points_percetnage(수면품질)이 상승하는 경향을 보입니다.</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: left;'>그렇다면 운동시간(active_minutes)가 증가할수록 sleep_points_percetnage(수면품질)이 하락하는 이유가 무엇일까요?</h3>",
    unsafe_allow_html=True
)