import streamlit as st
import pandas as pd
import tempfile
import cv2
import matplotlib.pyplot as plt
from calorie_estimator import analyze_image

plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.titlecolor": "white"
})

st.set_page_config(
    page_title="PlateCalc",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ PlateCalc")
st.markdown("AI-Powered Food Detection & Calorie Estimator")

uploaded_file=st.file_uploader("Upload a food image",type=["jpg","png","jpeg"])

if uploaded_file is not None :

    with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as tmp:
        
        tmp.write(uploaded_file.read())

        image_path=tmp.name
    
    with st.spinner("Analysing image...."):

        foods,total_calories,result=analyze_image(image_path)
    
    st.success("Analysis Complete!")

    st.metric(
        label="Total Calories",
        value=f"{total_calories:.1f} kcal"
    )

    st.subheader("Detected Foods")

    segmented=result.plot()
    segmented=cv2.cvtColor(segmented,cv2.COLOR_BGR2RGB)

    col1,col2=st.columns(2)

    with col1:

        st.image(image_path,caption="Original Image")
    
    with col2:

        st.image(segmented,caption="Detected Foods")
        

    if foods:

        df=pd.DataFrame(foods)

        df.index=range(1,len(df)+1)
        
        df= df.rename(
            columns={
            "name":"Food",
            "weight":"Weight (g)",
            "calories":"Calories",
            "confidence":"Confidence",
            "percentage":"Area (%)"
        })

        df = df.groupby("Food", as_index=False).agg({
            "Weight (g)": "sum",
            "Calories": "sum",
            "Confidence": "mean",
            "Area (%)": "sum"
        })

        if "Calories" not in df.columns:
            df["Calories"] = 0

        df["Weight (g)"] = df["Weight (g)"].round(1)
        df["Calories"] = df["Calories"].round(1)
        df["Confidence"] = (df["Confidence"]*100).round(1)
        df["Confidence"] = df["Confidence"].round(2)
        df["Area (%)"] = df["Area (%)"].round(2)

        chart_df = df.copy()
        df = df[
            [
            "Food",
            "Weight (g)",
            "Calories",
            "Confidence"
            ]]

        df = df.rename(columns={"Confidence": "Confidence (%)"})
        
        st.subheader("Nutrition Breakdown")

        st.dataframe(df,width="stretch")

        if df["Calories"].sum() > 0:

            fig1,ax1=plt.subplots(figsize=(6,4),facecolor="none")

            ax1.set_facecolor("none")
            ax1.bar(
                df["Food"],
                df["Calories"],
                width=0.6
                )
            
            ax1.grid(axis="y", alpha=0.3)
            ax1.tick_params(axis='x',rotation=45)
            ax1.set_ylabel("Calories (kcal)")
            ax1.set_title("Calories by Food Item")
            
            plt.tight_layout()

            fig2,ax2=plt.subplots(figsize=(5,5),facecolor="none")

            ax2.pie(
                chart_df["Area (%)"],
                labels=chart_df["Food"],
                autopct="%1.1f%%"
            )

            ax2.set_facecolor("none")

            ax2.set_title("Food Area Distribution")

            chart1, chart2 = st.columns(2)

            with chart1:
                st.subheader("Calories")
                st.pyplot(fig1)

            with chart2:
                st.subheader("Food Area")
                st.pyplot(fig2)