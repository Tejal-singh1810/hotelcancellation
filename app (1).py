
import numpy as np
import pandas as pd
import streamlit as st
import joblib

with open('final_model.joblib', 'rb') as file:
    model = joblib.load(file)

with open('transformer.joblib', 'rb') as file:
    transformer = joblib.load(file)

def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:,1][0]

    if pred>0.5:
        return f'This Booking is more likely to get cancelled with chances of {round(pred, 2)}'
    else:
        return f'This Booking is less likely to get Cancelled, chances {round(pred, 2)}'

def main():
    st.title('INN HOTEL GROUP')
    lt = st.text_input('Enter the Lead time in Days')
    mkt = (lambda x: 1 if x=='Online' else 0)(st.selectbox('How the booking was made?', ['Online', 'Offline']))
    price = st.number_input('Enter the price of the Room')
    adult = st.selectbox('How many Adults', [1,2,3,4])
    arr_m = st.slider('Select month', min_value=1, max_value=12, step=1)
    weekd_lambda = (lambda x: 0 if x=='Mon' else 
                                1 if x=='Tue' else 
                                2 if x=='Wed' else 
                                3 if x=='Thu' else 
                                4 if x=='Fri' else 
                                5 if x=='Sat' else 6)
    arr_w = weekd_lambda(st.selectbox('Weekday of Arrival', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))
    dep_w = weekd_lambda(st.selectbox('Weekday of Departure', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))
    weekn = st.text_input('Enter the no of week nights in stay')
    wkndn = st.text_input('Enter the no of weekend nights in stay')
    totan = weekn + wkndn
    park = (lambda x: 1 if x=='yes' else 0 )(st.selectbox('Parking Required ?', ['yes', 'no']))
    spcl = st.selectbox('How many special request have been made ?', [0,1,2,3,4,5])

    lt_t, price_t = transformer.transform([[lt, price]])[0]

    inp_list = [lt_t, spcl, price, adult, wkndn, park, weekn, mkt, arr_m, arr_w, totan, dep_w]

    if st.button('Predict'):
        response = prediction(inp_list)
        st.success(response)

if __name__ == '__main__':
    main()

    
    
