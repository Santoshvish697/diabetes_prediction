from math import perm
from isort import find_imports_in_code
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from Final_ML_Analysis_LR_DEC_RF_SVM import rf_classifier
from Final_ML_Analysis_LR_DEC_RF_SVM import dtree_classifier, log_regression, sv_classifier
from PIL import Image



# st.title("Dashboard")
st.set_page_config(
    page_title="Dashboard", page_icon="📊", initial_sidebar_state="expanded"
)

# def max_width():

#     max_width_str = f"max-width: 1400px;"
#     st.markdown(
#      f"""
#     <style>
#     .reportview-container .main .block-container{{
#         {max_width_str}
#     }}
#     </style>    
#     """,
#         unsafe_allow_html=True,
#     )

# max_width()
co1,co2,col3,col4 = st.columns([1,1,4,1])
with col3:
        components.html("""<html>
    <body>
    <div style = "background-color:red;margin:0 auto;width:100%;height:100px;text-align: center;padding:3rem;line-height:50px;">
    <h2 style= "margin:0 auto;color:white;position:relative;right:35px;">
    Prediction

    Positive</h2>
    </div>
    </body>
    </html>""", width=200, height=200)


with st.expander("👤- User", expanded=True):
    col1,col2,col3,col4,col5 = st.columns([2,2,2,2,3])
    with col1:
        st.button('Login History')
    with col2:
        st.button('Appointment History')
    with col3:
        st.button('Medicine & Food Intake')
    with col4:
        st.button('Medical Portfolio')
    with col5:
        st.button('Logout')

st.write(
    """
# 📊 Dashboard
"""
)

left_column,mid_column,right_column = st.columns([2.5,1,8])

if __name__ == "__main__":

    perm = 0
    with left_column:
        st.subheader("ML Analysis")
        # if (rf_classifier(perm) == 1):
        st.markdown("***Random Forest Classifier***")
        image_rfc = Image.open('pages/images/heat_rfc.png')
        st.image(image_rfc,caption = "Accuracy = 98.5%")
        # st.subheader("Random Forest Classifier") 
        print("RF SUCCESS!\n")
            
        # if (log_regression() == 1):
        st.markdown("***Logistic Regression***")
        image_logreg = Image.open('pages/images/log_reg.png')
        st.image(image_logreg, caption = "Accuracy = 78.25%")
        # st.subheader("Logistic Regression") 
        print("LOG REG SUCESS!\n")

    # if (sv_classifier() == 1):
        st.markdown("***SVC Classifier***")
        image_svc = Image.open('pages/images/svc.png')
        st.image(image_svc,caption = "Accuracy = 99.43%")
        # st.subheader("SVC Classifier") 
        print("SVM SUCCESS!")

        # if (dtree_classifier() == 1):
        st.write("***D-Tree Classifier***")
        image_dtree = Image.open("pages/images/heat_dtree.png")
        st.image(image_dtree,caption = "Accuracy = 97.5%")
        # st.subheader("DTREE Classifier")
        print("DTREE SUCCESS!")
    
    with right_column:
        perm = 1
        st.subheader("Effect of health parameters on Diabetes")
        # if rf_classifier(perm) == 1:
        image_fimp = Image.open('pages/images/f_imp.png')
        st.image(image_fimp)

        img_plot = Image.open('pages/images/plot.png')
        st.image(img_plot)
        print("SUCCESS!\n")