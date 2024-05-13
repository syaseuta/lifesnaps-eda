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

# ###모든 글자는 st.markdown(
#     "<h2 style='text-align: center;'>스트레스 관리와 수면의 질, 활동과의 상관관계</h2>"
#     "<h3 style='text-align: center;'>Top10 heatmap</h3>", 
#     unsafe_allow_html=True
# )###로 표현했음 
st.markdown(
    "<h1 style='text-align: center;'>Exertion_points_percentage(활동지수)는 무엇인가?</h1>", 
    unsafe_allow_html=True
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