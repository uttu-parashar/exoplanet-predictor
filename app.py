import streamlit as st
import numpy as np
import numpy
import pickle
import joblib


def welcome():
    return "Welcome All"

def Is_There_a_Exoplanet_in_orbit_of_that_star(raw_point):
    if len(raw_point) == 3197:
        # Outlier_detection and removing
        outlier_removed_point = []
        outlier_dict_path = r"outlier_history_dict.npy"
        outlier_history_dict = numpy.load(outlier_dict_path,allow_pickle='TRUE').item()
        for index in range(1,len(raw_point)+1):
            new_value = raw_point[index-1]
            if raw_point[index-1] < outlier_history_dict["FLUX."+str(index)]['percentile_1th']:
                new_value = outlier_history_dict["FLUX."+str(index)]['median']
            if raw_point[index-1] > outlier_history_dict["FLUX."+str(index)]['percentile_99th']:
                new_value = outlier_history_dict["FLUX."+str(index)]['median']
            outlier_removed_point.append(new_value)
        
        # Scalling point
        scalled_point = []
        scalling_history_dict_path = r"scalling_history_dict.npy"
        scalling_history_dict = numpy.load(scalling_history_dict_path,allow_pickle='TRUE').item()
        for index in range(1,len(raw_point)+1):
            feature_value = raw_point[index-1]
            min_feature = scalling_history_dict["FLUX."+str(index)]['min_feature']
            max_feature = scalling_history_dict["FLUX."+str(index)]['max_feature']
            scalled_value = ((feature_value - min_feature) / (max_feature - min_feature))
            scalled_point.append(scalled_value)
        
        # Loading Model for Prediction
        model_path = r"Best_performance_model.pkl"
        model = joblib.load(model_path)
        
    
        # predicting point
    
        prediction = model.predict(np.array(scalled_point).reshape(1, -1))
    
        if prediction == 0:
            result_ = str("""This is a Negative Point. \n It means there is not any planet in orbit of that Star.""")
        else :
            result_ = str("""YES.. This is a Positive Point. \n It means there is atleast one planet in orbit of that Star.""") 
        return result_
    else:
        result_ = str("Please Give Right Raw Input to Predict..")
        return result_
    
def what_to_predict_(what_to_predict):
    if what_to_predict == "point_pos" :
        point_to_predict = np.load('sample_points.npy',allow_pickle='TRUE').item()["point_positive_class"]
        prediction = Is_There_a_Exoplanet_in_orbit_of_that_star(point_to_predict)
        return prediction
    if what_to_predict == "point_neg" :
        point_to_predict = np.load('sample_points.npy',allow_pickle='TRUE').item()["point_negative_class"]
        prediction = Is_There_a_Exoplanet_in_orbit_of_that_star(point_to_predict)
        return prediction
    we_got_a_real_point = what_to_predict
    if len(list(we_got_a_real_point.split(","))) == 3197:
        point_to_predict = [float(val) for val in list(we_got_a_real_point.split(","))]
        prediction = Is_There_a_Exoplanet_in_orbit_of_that_star(point_to_predict)
        return prediction
        
    return "Please Give the right Input."

def main():
    st.title("You are here. That means you also want to find a Exoplanet in Deep Space. Okeyy..!! :), That's Great to hear. Let's Do it..")
    html_temp = """
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;">Exoplanet-Finder Ml-Web App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Value = st.text_input("Give Intensity values of light coming from that star. or point_pos or point_neg")
    output=""
    if st.button("is there a exoplanet.?"):
        output=what_to_predict_(Value)
    st.success(output)
    
    
    
    
    if st.button("Blog Link"):
        st.text("https://medium.datadriveninvestor.com/lets-find-planets-beyond-our-solar-system-milky-way-galaxy-with-the-help-of-905dcfc95d3d")
    if st.button("Linked-In Profile Link"):
        st.text("https://www.linkedin.com/in/utkarsh-parashar-8529641a0/")
        
if __name__=='__main__':
    main()
