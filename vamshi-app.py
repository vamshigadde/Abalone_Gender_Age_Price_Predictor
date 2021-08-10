import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def main():
    st.title("""ABALONE AGE AND PRICE PREDICTOR""")
    
    activities = ["Select","Exploratory Data Analysis","Exploratory Data Visualization","Prediction Gender","Prediction Age","Prediction Price"]
    choice = st.sidebar.selectbox("Select Activities",activities)


    if choice == 'Select':
        st.image("abaphoto.jpg")
        st.write('''Abalone is a rich nutritious food resource in the many parts of the world.
                    The economic value of abalone is positively correlated with its age. However, determining the age of
                    abalone is a cumbersome as well as expensive process which increases the cost and limits its
                    popularity. This Analysis proposes very simple ways to determine the age of abalones using
                    econometric methods to reduce the costs of producers as well as consumers.''')



        st.write('''Predicting the age of abalone from physical measurements. The age of abalone
                    is determined by cutting the shell through the cone, staining it, and counting
                    the number of rings through a microscope -- a boring and time-consuming task.
                    Other measurements, which are easier to obtain, are used to predict the age.''')


        

    if choice == 'Exploratory Data Analysis':
        st.subheader("Exploratory Data Analysis")
        st.image("edae.jpg")

        data = st.file_uploader("Upload a Dataset",type=['csv','txt'])
        if data is not None :
            df = pd.read_csv(data)
            df.drop('Unnamed: 0',axis=1,inplace=True)
            st.dataframe(df.head())


            if st.checkbox("Show Shape"):
                st.write(df.shape)

##            if st.checkbox("Check Info"):
##                st.write(df.info())

            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Descriptive Statistics"):
                st.write(df.describe())

            if st.checkbox("See Null Values"):
                st.write(df.isna().sum())

            if st.checkbox("Correlation"):
                st.write(df.corr())

            

            

          


    elif choice == 'Exploratory Data Visualization':
        st.subheader("Exploratory Data Visualization")
        st.image("eda.jpg")
        data = st.file_uploader("Upload a Dataset",type=['csv','txt'])
        if data is not None:
            df = pd.read_csv(data)
            df.drop('Unnamed: 0',axis=1,inplace=True)
            st.dataframe(df.head())
            df['age'] = df['Rings']+1.5


           
            if st.checkbox("Countplot"):
                fig=plt.figure()
                st.write(sns.countplot(x='Gender',data=df,palette='YlOrRd'))
                st.pyplot()
           

            if st.checkbox("Correlation Plot(Matplotlib)"):
                
                plt.matshow(df.corr(),fignum=None)
                st.pyplot()

##            if st.checkbox("Histogram"):
##                fig=plt.figure()
##                st.write(df.hist(figsize=(20,10),grid=False,bins=30))
##                st.pyplot(fig)

            if st.checkbox("Correlation Plot(Seaborn)"):
                fig=plt.figure()
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot(fig)

            if st.checkbox("3D Plot (Diameter,Rings,Age)"):
                fig = plt.figure()
                ax= plt.axes(projection='3d')
                ax.set_xlabel('Diameter (mm)')
                ax.set_ylabel(' Rings')
                ax.set_zlabel('age')

                st.write(ax.scatter3D(df['Diameter'],df['Rings'],df['age'],c='blue'))
                st.pyplot(fig)


            if st.checkbox("3D Plot (Height,Length,Rings)"):
                fig = plt.figure()
                ax= plt.axes(projection='3d')
                ax.set_xlabel('Height (mm)')
                ax.set_ylabel('Length (mm)')
                ax.set_zlabel('Number of rings')

                st.write(ax.scatter3D(df['Height'],df['Length'],df['Rings'],c='red'))
                st.pyplot(fig)


            if st.checkbox("3D Plot (Whole Weight,Shell Weight,Age)"):
                fig = plt.figure()
                ax= plt.axes(projection='3d')
                ax.set_xlabel('Whole Weight')
                ax.set_ylabel('Shell Weight')
                ax.set_zlabel('Age')

                st.write(ax.scatter3D(df['Whole weight'],df['Shell weight'],df['age'],c='magenta'))
                st.pyplot(fig)

            if st.checkbox("3D Plot (Viscera weight,Length,Rings)"):
                fig = plt.figure()
                ax= plt.axes(projection='3d')
                ax.set_xlabel('Viscera weight')
                ax.set_ylabel('Length')
                ax.set_zlabel('Rings')

                st.write(ax.scatter3D(df['Viscera weight'],df['Length'],df['Rings'],c='magenta'))
                st.pyplot(fig)

            

            if st.checkbox("Swarmplot"):
                fig=plt.figure()
                st.write(sns.swarmplot(x='Gender',y='age',data=df,hue='Gender',palette=['green','crimson','Yellow']))
                st.pyplot(fig)

            if st.checkbox('Violinplot'):
                fig=plt.figure()
                st.write(sns.violinplot(x='Gender',y='age',data=df))
                st.pyplot(fig)

##            if st.checkbox('Jointplot'):
##                fig=plt.figure()
##                st.write(sns.jointplot(x='Length',y='age',data=df,kind='reg'))
##                st.pyplot(fig)
##
##            if st.checkbox('LMPlot'):
##                fig=plt.figure()
##                st.write(sns.lmplot(x='Diameter',y='age',data=df,col='Gender',palette='crimson'))
##                st.pyplot(fig)

            

##
##            if st.checkbox("Pairplot",value=True):
##                fig=plt.figure()
##                st.write(sns.pairplot(df))
##                st.pyplot(fig)



    elif choice == 'Prediction Gender':
        st.subheader("Prediction Gender")
        data = st.file_uploader("Upload a Dataset",type=['csv','txt'])

        df = pd.read_csv(data)
        df.drop('Unnamed: 0',axis=1,inplace=True)
        length = st.slider("Select Length",0.0000,1.0000)
        diameter = st.slider("Select Diameter",0.0000,1.0000)
        height = st.slider("Select Height",0.0000,1.0000)
        ww = st.slider("Select Whole Weight",0.0000,1.0000)
        sw = st.slider("Select Shucked Weight",0.0000,1.0000)
        vw = st.slider("Select Viscera Weight",0.0000,1.0000)
        shw = st.slider("Select Shell Weight",0.0000,1.0000)

        
        results = [length,diameter,height,ww,sw,vw,shw]
        displayed_results = [length,diameter,height,ww,sw,vw,shw]
        prettified_results = {'length':length,'diameter':diameter,'height':height,'ww':ww,'sw':sw,'vw':vw,'shw':shw}
        sample_data = np.array(results).reshape(1,-1)
        st.info(results)
        st.json(prettified_results)
        st.subheader("Prediction Aspect")
        x=df.iloc[:,:-3]
        y=df.iloc[:,-2]
        predictor = LogisticRegression()
        predictor.fit(x,y)
        prediction = predictor.predict(sample_data)[0]
        if st.button('Predict'):
            st.success(f"Your Predicted Age is {prediction}")

        



    elif choice == 'Prediction Age':
        st.subheader("Prediction Age")
        st.image("predage.jpg")

        data = st.file_uploader("Upload a Dataset",type=['csv','txt'])

        df = pd.read_csv(data)
        df.drop('Unnamed: 0',axis=1,inplace=True)
        
        
        length = st.slider("Select Length",0.0000,1.0000)
        diameter = st.slider("Select Diameter",0.0000,1.0000)
        height = st.slider("Select Height",0.0000,1.0000)
        ww = st.slider("Select Whole Weight",0.0000,1.0000)
        sw = st.slider("Select Shucked Weight",0.0000,1.0000)
        vw = st.slider("Select Viscera Weight",0.0000,1.0000)
        shw = st.slider("Select Shell Weight",0.0000,1.0000)

        results = [length,diameter,height,ww,sw,vw,shw]
        displayed_results = [length,diameter,height,ww,sw,vw,shw]
        prettified_results = {'length':length,'diameter':diameter,'height':height,'ww':ww,'sw':sw,'vw':vw,'shw':shw}
        sample_data = np.array(results).reshape(1,-1)
        st.info(results)
        st.json(prettified_results)
        st.subheader("Prediction Aspect")
        x=df.iloc[:,:-3]
        y=df.iloc[:,-1]

        predictor = RandomForestClassifier(bootstrap=True,max_depth=80,max_features =2,min_samples_leaf= 5,min_samples_split=30,n_estimators= 100,random_state=8)
        predictor.fit(x,y)
        prediction = predictor.predict(sample_data)[0]
        if st.button('Predict'):
            st.success(f"Your Predicted Age is {prediction}")


    elif choice == 'Prediction Price':
        st.subheader("Prediction Price")
        st.image("predprice.png")
        data = st.file_uploader("Upload a Dataset",type=['csv','txt'])
        df = pd.read_csv(data)
        df.drop('Unnamed: 0',axis=1,inplace=True)
        st.dataframe(df.head())

        x=np.array(df['Age']).reshape(-1,1)
        predictor = LinearRegression()
        predictor.fit(x,np.array(df['Price']))


        val=st.number_input("Enter Age",1,20,step=1)
        val=np.array(val).reshape(1,-1)
        

        prediction = predictor.predict(val)[0]
        if st.button('Predict'):
            st.success(f"Your Predicted Price of Abalone is {round(prediction)}$")
        
if __name__=='__main__':
    main()
                    
                                                         
