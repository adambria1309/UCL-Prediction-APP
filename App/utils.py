import pandas as pd
import streamlit as st

@st.cache(suppress_st_warning=True)
def read_data(path):
    return pd.read_csv(path)

def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px; color: white'>
        UCL MATCH PREDICTION
        </h1>
    """, unsafe_allow_html=True
    )
    
    st.caption("""
        <p style='text-align: center; font-size: 16px;color: white'>
        Developer: <a href='https://github.com/adambria1309'>Adam Alexander Bria</a>
        </p>
    """, unsafe_allow_html=True
    )
    
    st.markdown("""
        <p style='font-size: 20px; text-align: justify; color: white'>
        Prediction result is based on the data of UCL knockout stages from 2002/2003 season to 2021/2022 season. Since Club Brugge and Eintracht Frankfurt had never played in a UCL knockout stages before, this model cannot comprehend to predict their match result.
        </p>    
    """, unsafe_allow_html=True
    )

    st.markdown("""
        <p style='font-size: 20px; text-align: justify; color: white'>
        Please note that there are many factors that are not included in the prediction model. The actual match results could be very different.
        </p>    
    """, unsafe_allow_html=True
    )

def body():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    st.markdown("""
        <br/>
        <h5 style='text-align: left; margin-bottom: -35px; color: white'>
        Select Home Team
        </h5>
    """, unsafe_allow_html=True
    )

    team1 = st.selectbox(
        "",["Paris S-G","Milan","Bayern Munich","Tottenham","Benfica","Dortmund",
        "Chelsea","Liverpool","Real Madrid","Napoli"],key="team1"
    )

    st.markdown("""
        <br/>
        <h5 style='text-align: left; margin-bottom: -35px; color: white'>
        Select Away Team
        </h5>
    """, unsafe_allow_html=True
    )
    team2 = st.selectbox(
        " ",["Paris S-G","Milan","Bayern Munich","Tottenham","Benfica","Dortmund",
        "Chelsea","Liverpool","Real Madrid","Napoli"],key="team2"
    )

    
    predict = st.button("Predict Match Result")

    if predict:
        df = read_data('./Data/ucl2.csv')

        # fit the regressor with x and y data
        categorical_cols = ['Team1','Team2']
        df1 = df.drop(columns = ['Goal_Team_2','Hasil'])
        le = LabelEncoder()
        df1[categorical_cols] = df1[categorical_cols].apply(lambda col: le.fit_transform(col))

        #Get independent variables
        X1 = df1.drop(columns = ['Goal_Team_1'])
        X1 = X1.loc[:,categorical_cols]

        #Get dependent variable
        Goal1= df1.loc[:,'Goal_Team_1']

        #Turn dependent variable into an array
        y1 = Goal1.values

        # Split dataset into training set and test set
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2)
        
        # fit model 1
        regc1 = RandomForestClassifier(n_estimators=8)
        regc1.fit(X_train1, y_train1)

        # fit the regressor with x and y data 2
        categorical_cols = ['Team1','Team2']
        df2 = df.drop(columns = ['Goal_Team_1','Hasil'])
        le = LabelEncoder()
        df2[categorical_cols] = df2[categorical_cols].apply(lambda col: le.fit_transform(col))

        #Get independent variables
        X2 = df2.drop(columns = ['Goal_Team_2'])
        X2 = X2.loc[:,categorical_cols]

        #Get dependent variable
        Goal2= df2.loc[:,'Goal_Team_2']

        #Turn dependent variable into an array
        y2 = Goal2.values

        # Split dataset into training set and test set
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)

        # fit model 1
        regc2 = RandomForestClassifier(n_estimators=6)
        regc2.fit(X_train2, y_train2)

        ## predict goal for team 1
        test = pd.DataFrame({'Team1':[team1],
                            'Team2':[team2]})

        test[['Team1','Team2']] = test[['Team1','Team2']].apply(lambda t: le.fit_transform(t))
        pred1 = regc1.predict(test.to_numpy())
        pred2 = regc2.predict(test.to_numpy())

        # y_pred1 = regc1.predict(X_test1)
        # y_pred2 = regc2.predict(X_test2)

        prob1 = max(max(regc1.predict_proba(test)))*100
        prob2 = max(max(regc2.predict_proba(test)))*100

        if pred1==pred2:
            st.markdown(f"""
            <h4 style='text-align: center; color: white;'>
                DRAW!
            </h4><br/>
            <h6 style='text-align: center; color: white;'>   
                {team1}:{pred1} VS {team2}:{pred2} <br/>
                Probability: {round((prob1+prob2)/2,2)}%
            </h6>
            """,unsafe_allow_html=True)
        elif pred1<pred2:
            st.markdown(f"""
            <h4 style='text-align: center; color: white;'>
                {team2} WIN! 
            </h4><br/>
            <h6 style='text-align: center; color: white;'>    
                {team1}{pred1} VS {team2}{pred2} </br>
                Probability: {round((prob2),2)}%
            </h6>
            """,unsafe_allow_html=True)
        elif pred1>pred2:
            st.markdown(f"""
            <h4 style='text-align: center; color: white;'>
                {team1} WIN! 
            </h4><br/>
            <h6 style='text-align: center; color: white;'>
                {team1}{pred1} VS {team2}{pred2} <br/>
                Probability: {round((prob1),2)}%
            </h6>
            """,unsafe_allow_html=True)


     
