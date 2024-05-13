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

#1.4. 상관관계 Top 10을 통해 왜 스트레스, 수면, 활동(운동)에 주목해야되는지 설명 [Heatmap] [코드 by 인웅]


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
folder_path = os.path.join("../output_data/")
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

numeric_columns = daily_sema.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_columns.corr()
st.markdown(
    "<h3>스트레스 관리와 수면품질, 활동과의 관계</h3>"
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
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(top_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
plt.title('Top 10 Correlation Heatmap (for numeric columns)')
st.pyplot(fig)
