import streamlit as st
import multilanDetect as md
st.set_page_config(
    page_title='小小語言試劑',
    layout='wide'
)
import boardPredict as bp

want_to_complete = st.sidebar.selectbox('你想完成的任務...', ['PTT標題分類' ,'多國語言偵測器']) # , 'NLP小任務'
# title and description
st.write("""
# 小小語言試劑
""")
st.markdown('資料取自[PTT](https://term.ptt.cc/)及[kaggle](https://www.kaggle.com/datasets/basilb2s/language-detection)。')

if want_to_complete == 'PTT標題分類':
    st.write("""
    ### PTT標題分類
    """)
    options = st.sidebar.multiselect('你想選擇的看板 (可複選)', ['八卦版', '寶可夢版', '美劇版','笑話板','軟工版'], default=["八卦版"])
    select_model = st.sidebar.selectbox('你想選擇的模型', ['貝氏', 'XGBoost'])
    # search bar
    query = st.text_input("輸入一段話：", placeholder="徵一隻6v異國皮卡丘")
    st.write('Board you selected:', options, 'and press the button.')
    
    result=""
    if st.button("Predict"):
        result=bp.train_predict(options, query, select_model)
        st.success('這段文字的風格類似 {}'.format(result))

# if want_to_complete == 'NLP小任務':
#     st.write("""
#     ### NLP小任務
#     """)

if want_to_complete == '多國語言偵測器':
    st.write("""
    ### 多國語言偵測器
    17 different languages:  English, Portuguese, French, Greek, 
    Dutch, Spanish, Japanese, Russian, Danish, Italian, Turkish, 
    Swedish, Arabic, Malayalam, Hindi, Tamil, Telugu
    """)
    # search bar
    query = st.text_input("輸入一段話：", placeholder="Καλώς ήλθατε στο blog του PY Tsai")

    result=""
    if st.button("Predict"):
        result=md.load_predict(query)
        st.success('The given text is written in {}'.format(result))