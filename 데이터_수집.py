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
import io

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(page_title="데이터 수집")
    st.markdown(
        "<h1 style='text-align: center;'>Kaggle Fitbit Sleep EDA Project</h1><br><br>",
        unsafe_allow_html=True
    )

    st.markdown("""
        <h3>주제</h3>
        <style>
        .text {
            font-size: 14px;
        }
        </style>
        <div class="text">
        fitbit 기기로 수집한 생체 데이터를 바탕으로 분석한 수면 품질에 미치는 요인에 대한 분석
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<br><h3>데이터 수집</h3>",
        unsafe_allow_html=True
    )

    st.markdown("""
        <h4>데이터 수집 개요</h4>
        <style>
        .text {
            font-size: 14px;
        }
        </style>
        <div class="text">
        1. 기간 : 약 4개월<br>
        2. 참가자 수 : 71명<br>
        3. 방법 : 스마트워치를 통한 기록과 설문조사<br>
        4. 데이터 : 신체 활동 패턴, 수면, 스트레스 및 전반적인 건강과 행동 및 심리적 패턴<br>
        5. 제한 사항 : 참가자들은 평소와 같이 자연스럽게 행동하도록 지시받았고, 실험실 조건이나 제한은 별도로 부여되지 않았음<br>
        6. 모니터링 : 참가자의 참여를 유지하기 위해 주간 알림 및 후속 조치<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <h4>참가자</h4>
    <style>
    .text {
        font-size: 14px;
    }
    </style>
    <div class="text">
    1. 국적 분포 : 스웨덴 24명, 이탈리아 10명, 그리스 25명, 키프로스 12명<br>
    2. 성별 : 남성 42명, 여성 29명<br>
    3. 연령<br>
	    - 모집 당시 최소 18세 이상<br>
        - 두 명을 제외하고 모두 나이를 제공<br>
        - 30세 미만과 30세 이상이 절반씩 분포(익명성이 보장되도록 범위가 정의됨)<br>
    4. 모집 경로 : 샘플링 또는 대학 메일링 리스트에 대한 자원봉사 전화를 통해 모집<br>
    5. 참가자에게는 참여에 대한 금전적 또는 기타 인센티브가 제공되지 않았음<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4>데이터 종류</h4>
    <style>
    .text {
        font-size: 14px;
    }
    </style>
    <div class="text">
    1. Fitbit : 플래그십 스마트워치에서 데이터 감지<br>
    2. SEMA3 : 참가자의 일일 목표, 기분 및 상황을 추출하기 위한 생태학적 순간 평가<br>
    3. 설문조사 : 성격 및 불안 점수와 같은 평가가 포함된 설문조사 데이터<br>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()


@st.cache_resource
def load_data():
    folder_path = os.path.join("./output_data/")
    daily = pd.read_csv(f'{folder_path}' + 'daily_fitbit_sema_df_unprocessed.csv')
    hourly = pd.read_csv(f'{folder_path}' + 'hourly_fitbit_sema_df_unprocessed.csv')
    return daily, hourly


daily_sema, hourly_sema = load_data()

numeric_columns = daily_sema.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_columns.corr()
st.markdown(
    "<br><h3>스트레스 관리와 수면품질, 활동과의 관계</h3>"
    "<h4>Top10 heatmap</h4>", 
    unsafe_allow_html=True
)
st.markdown("""
<style>
.text {
    font-size: 14px;
}
</style>
<div class="text">
이 그래프는 가장 상관관계가 높은 상위 10개의 열을 선택하여 히트맵을 그렸습니다.
이를 통해 스트레스 관리 능력과 활동, 수면 간의 연결성을 짐작해볼 수 있습니다.
stress_score와 exertion_points_percentage의 상관계수가 0.96이고
stress_score와 sleep_points_percentage의 상관계수가 0.95임을 고려해볼 때
stress_score 즉 스트레스 관리 능력이 높을 수록 운동비율과 수면의 질이 향상됨을 예상할 수 있습니다.<br><br>
</div>
""", unsafe_allow_html=True)
# 가장 상관관계가 높은 상위 10개의 열 찾기
top_correlation_columns = correlation_matrix.abs().stack().nlargest(10).index
top_correlation_matrix = correlation_matrix.loc[top_correlation_columns.get_level_values(0), top_correlation_columns.get_level_values(1)]

# 히트맵 그리기
@st.cache_resource
def create_heatmap(top_correlation_matrix):
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(top_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title('Top 10 Correlation Heatmap (for numeric columns)')

    # 그래프 요소들을 조정하여 잘림 방지
    plt.tight_layout()

    # 이미지 데이터 생성
    image_data = io.BytesIO()
    fig.savefig(image_data, format='png')
    image_data.seek(0)

    return image_data


# 이미지 데이터를 캐시하고 필요할 때 로드하여 표시
heatmap_image_data = create_heatmap(top_correlation_matrix)
st.image(heatmap_image_data, use_column_width=True)
