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
import seaborn as sns
from datetime import * 
from functools import reduce
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import streamlit as st

from 데이터_수집 import daily_sema

# 필요한 컬럼 선택
df_subset = daily_sema[['id', 'sleep_duration', 'sleep_efficiency', 'sleep_points_percentage', 'sleep_deep_ratio', 'sleep_wake_ratio',
                        'sleep_light_ratio', 'sleep_rem_ratio']]

# 결측값 제거
df_subset.dropna(subset=['sleep_duration'], inplace=True)

# 밀리초를 시간으로 변환하는 함수
def milliseconds_to_hours(milliseconds):
    hours = milliseconds / (1000 * 60 * 60)
    return hours

# 'sleep_duration' 컬럼의 값을 변환하여 새로운 컬럼에 저장
df_subset['sleep_duration_hours'] = df_subset['sleep_duration'].apply(milliseconds_to_hours)

# sleep_points_percentage 값을 100배하여 퍼센트로 표시
df_subset['sleep_points_percentage'] = df_subset['sleep_points_percentage'] * 100

# ID 별로 그룹화하여 평균값 계산
grouped_mean = df_subset.groupby('id').mean()

st.markdown(
    "<h1 style='text-align: center;'>Kaggle Fitbit Sleep EDA Project</h1><br><br>", 
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center;'>수면 효율성 및 수면 포인트 분석</h2>", 
    unsafe_allow_html=True
)
st.markdown(    
    "<h3>수면 평균 시간 및 수면 포인트 비율 관계 분석</h3>", 
    unsafe_allow_html=True
)

@st.cache_resource
def plot_regression_graph(data, x_column, y_column, title='Regression Plot', x_label=None, y_label=None):
    plt.figure(figsize=(8, 6))
    sns.regplot(data=data, x=x_column, y=y_column)
    plt.xlim(6, 10)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # BytesIO 객체 생성
    plot_image = io.BytesIO()
    
    # 그래프를 이미지 파일로 저장
    plt.savefig(plot_image, format='png')
    
    # BytesIO 객체의 파일 포인터를 처음으로 되돌림
    plot_image.seek(0)
    
    # BytesIO 객체의 내용 반환
    return plot_image.getvalue()

# 그래프 플로팅
plot_image_bytes = plot_regression_graph(grouped_mean, x_column='sleep_duration_hours', y_column='sleep_points_percentage', title='Sleep Duration vs Sleep Points Percentage', x_label='Sleep Duration (hours)', y_label='Sleep Points Percentage')

# Streamlit에 그래프를 표시
st.image(plot_image_bytes, use_column_width=True)

st.markdown("""**수면 시간과 수면 포인트 비율 비교**    
            → 수면 시간은 **6~8시간**으로 분포 되어있으며  
            → 수면 포인트 비율성은 **30~80퍼센트**로 고르게 분포되어 있음    
            → 전체적으로 봤을때 **양의 상관성**을 띈다 BUT 하지만 분포도로 따졌을때 **6~8시간** 수면 시간이 제일 **높은 비율**이 나옴""")

st.markdown(
    "<h2>수면 효율성 및 수면 포인트 비율 관계 분석</h2>", 
    unsafe_allow_html=True
)

@st.cache_resource
def plot_regression_graph(data, x_column, y_column, title='Regression Plot', x_label=None, y_label=None):
    plt.figure(figsize=(8, 6))
    sns.regplot(data=data, x=x_column, y=y_column)
    plt.xlim(90, 97)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # BytesIO 객체 생성
    plot_image = io.BytesIO()
    
    # 그래프를 이미지 파일로 저장
    plt.savefig(plot_image, format='png')
    
    # BytesIO 객체의 파일 포인터를 처음으로 되돌림
    plot_image.seek(0)
    
    # BytesIO 객체의 내용 반환
    return plot_image.getvalue()

# 그래프 플로팅
plot_image_bytes = plot_regression_graph(grouped_mean, x_column='sleep_efficiency', y_column='sleep_points_percentage', title='Sleep Points Percentage vs Sleep Efficiency', x_label='Sleep Efficiency', y_label='Sleep Points Percentage')

# Streamlit에 그래프를 표시
st.image(plot_image_bytes, use_column_width=True)


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
    "<h2 style='text-align: left;'>Exertion_points_percentage(활동지수)는 무엇인가?</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: left;'>Exertion_points_percentage(활동지수)에 영향을 미치는 요인</h3>"
    "<h4 style='text-align: left;'>Heatmap Top 10</h4>", 
    unsafe_allow_html=True
)

exertion_selcted_columns = ['exertion_points_percentage', 'stress_score', 'calories', 'distance', 'lightly_active_minutes', 'moderately_active_minutes', 'very_active_minutes', 'total_active_minutes', 'sedentary_minutes', 'resting_hr']
exertion_corr = daily_sema[exertion_selcted_columns].corr()

@st.cache_resource
def plot_heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', title='Heatmap'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data=data, annot=annot, fmt=fmt, cmap=cmap)
    plt.title(title)

    # 그래프 요소들을 조정하여 잘림 방지
    plt.tight_layout()

    # BytesIO 객체 생성
    plot_image = io.BytesIO()
    
    # 그래프를 이미지 파일로 저장
    plt.savefig(plot_image, format='png')
    
    # BytesIO 객체의 파일 포인터를 처음으로 되돌림
    plot_image.seek(0)
    
    # BytesIO 객체의 내용 반환
    return plot_image.getvalue()

# heatmap 플로팅
heatmap_image_bytes = plot_heatmap(data=exertion_corr, annot=True, fmt='.2f', cmap='coolwarm', title='Exertion Correlation Heatmap')

# Streamlit에 heatmap 표시
st.image(heatmap_image_bytes, use_column_width=True)

st.markdown(
    "<h3 style='text-align: left;'>exertion_points_percentage에 영향을 미치는 요인 8개</h3>", 
    unsafe_allow_html=True
)

@st.cache_resource
def plot_subplot_lineplots_cached(data, y_columns, x_column='exertion_points_percentage', figsize=(18, 14), title=None):
    fig, axs = plt.subplots(3, 3, figsize=figsize)
    axs[-1, -1].remove()  # 마지막 subplot 삭제

    # 각 subplot에 대해 line plot 그리기
    for i, ax in enumerate(axs.flat):
        # y축 설정
        if i < len(y_columns):
            y_column = y_columns[i]

            # line plot 그리기
            sns.lineplot(x=x_column, y=y_column, data=data, ax=ax)

            # x축과 y축 레이블 설정
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

    # 전체 그래프의 제목 설정
    if title:
        plt.suptitle(title)

    # 레이아웃 조정
    plt.tight_layout()

    # BytesIO 객체 생성
    plot_image = io.BytesIO()

    # 그래프를 이미지 파일로 저장
    plt.savefig(plot_image, format='png')

    # BytesIO 객체의 파일 포인터를 처음으로 되돌림
    plot_image.seek(0)

    # BytesIO 객체의 내용 반환
    return plot_image.getvalue()

# subplot에 그래프 플로팅 (캐싱 사용)
subplot_image_bytes = plot_subplot_lineplots_cached(data=daily_sema, y_columns=['calories', 'distance', 'total_active_minutes', 
                                                                        'lightly_active_minutes', 'moderately_active_minutes', 
                                                                        'very_active_minutes', 'sedentary_minutes', 'resting_hr'], 
                                              x_column='exertion_points_percentage', 
                                              figsize=(18, 14), 
                                              title='Exertion Points Percentage vs Various Activities')

# Streamlit에 subplot 그래프 표시
st.image(subplot_image_bytes, use_column_width=True)
    

st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
calories, distance, total_active_minutes 처럼 활동과 관련된 지표가 상승할수록 exertion_points_percentage는 상승하며, 
반대로 sedentary_minutes와 resting_hr처럼 휴식 지표가 상승할수록 exertion_points_percentage가 하락한다.
따라서 exertion_points_percentage가 활동지수로서 신뢰할 만한 지표인 것을 확인할 수 있다.<br><br>
</div>
""", unsafe_allow_html=True)