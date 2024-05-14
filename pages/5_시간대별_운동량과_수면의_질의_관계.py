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
import io
import streamlit as st
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from 데이터_수집 import daily_sema
from 데이터_수집 import hourly_sema

st.set_page_config(page_title="시간대별 운동량과 수면의 질의 관계")
st.markdown(
    "<h1 style='text-align: center;'>Kaggle Fitbit Sleep EDA Project</h1><br><br>", 
    unsafe_allow_html=True
)

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
    "<h4 style='text-align: left;'>평균 칼로리 소모량 상위 25%인 그룹과 하위 25%인 그룹을 추출하여\
        시간대별 칼로리 소모량을 추적함</h4>",
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
    "<h3 style='text-align: left;'>평균 칼로리 소모량 상위 25%인 그룹의 시간대별 칼로리 소모량 곡선\
        </h3>",
    unsafe_allow_html=True
)

@st.cache_resource
def plot_multiple_lineplots_upper_25(data, ids, x, y):
    # Create a BytesIO object to store the image
    buffer = io.BytesIO()
    
    plt.figure(figsize=(30, 20))
    i = 1
    for id in ids:
        plt.subplot(3, 4, i)
        sns.lineplot(data=data.loc[data['id'] == id], x=x, y=y)
        plt.title(id)
        i += 1

    plt.tight_layout()
    plt.savefig(buffer, format='png')  # Save plot to BytesIO object in PNG format
    plt.close()  # Close the plot to free up memory

    # Return the BytesIO object containing the image data
    return buffer.getvalue()

plot_image1 = plot_multiple_lineplots_upper_25(hourly_sema, lower_25_ids, 'hour', 'calories')

st.image(plot_image1, use_column_width=True)

st.markdown(
    "<h3 style='text-align: left;'>평균 칼로리 소모량 하위 25%인 그룹의 시간대별 칼로리 소모량 곡선\
        </h3>",
    unsafe_allow_html=True
)

@st.cache_resource
def plot_multiple_lineplots_lower_25(data, ids, x, y):
    # Create a BytesIO object to store the image
    buffer = io.BytesIO()
    
    plt.figure(figsize=(30, 20))
    i = 1
    for id in ids:
        plt.subplot(3, 4, i)
        sns.lineplot(data=data.loc[data['id'] == id], x=x, y=y)
        plt.title(id)
        i += 1

    plt.tight_layout()
    plt.savefig(buffer, format='png')  # Save plot to BytesIO object in PNG format
    plt.close()  # Close the plot to free up memory

    # Return the BytesIO object containing the image data
    return buffer.getvalue()

# 예시로 함수를 호출하는 방법
plot_image2 = plot_multiple_lineplots_lower_25(hourly_sema, lower_25_ids, 'hour', 'calories')

st.image(plot_image2, use_column_width=True)

st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
상위 25%인 그룹의 저녁 시간대(7~9시 사이) 운동량이 하위 25%에 비해 월등이 높다.<br><br>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align: center;'>상위 25%인 그룹(파란색)과 하위 25%인 그룹(오렌지색)의 시간대별 운동량.\
        </h3>",
    unsafe_allow_html=True
)

combined_data_upper = pd.concat([hourly_sema.loc[hourly_sema['id'] == id] for id in upper_25_ids])
combined_data_lower = pd.concat([hourly_sema.loc[hourly_sema['id'] == id] for id in lower_25_ids])

# kdeplot으로 데이터의 분포 시각화
@st.cache_resource
def plot_combined_kde_plot(data_upper, data_lower, x, y):
    # Create a BytesIO object to store the image
    buffer = io.BytesIO()
    
    plt.figure(figsize=(10, 10))
    sns.kdeplot(data=data_upper, x=x, y=y, color='blue', label='upper 25%', shade=True)
    sns.kdeplot(data=data_lower, x=x, y=y, color='orange', label='lower 25%', shade=True)
    plt.ylim(0, 400)
    plt.title('Combined KDE Plot for lower 25% ids vs upper 25% ids')
    plt.legend(loc='upper left', fontsize='large')

    plt.tight_layout()
    plt.savefig(buffer, format='png')  # Save plot to BytesIO object in PNG format
    plt.close()  # Close the plot to free up memory

    # Return the BytesIO object containing the image data
    return buffer.getvalue()

# 예시로 함수를 호출하는 방법
plot_image3 = plot_combined_kde_plot(combined_data_upper, combined_data_lower, 'hour', 'calories')

st.image(plot_image3, use_column_width=True)

st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
상위 25%인 그룹(파란색)은 마치 낙타처럼 저녁 시간대의 운동량이 급격하게 상승한 모습을 볼 수 있다. 
즉, 저녁 시간대의 운동은 수면 품질에 악영향을 미칠 수 있다.<br><br>
</div>
""", unsafe_allow_html=True)