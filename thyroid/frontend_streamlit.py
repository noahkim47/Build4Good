import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

country_encoding = {
    "Russia": 0.10013118,
    "Germany": 0.04963539,
    "Nigeria": 0.15006747,
    "India": 0.19980159,
    "UK": 0.005003503,
    "South Korea": 0.07036029,
    "Brazil": 0.10067657,
    "China": 0.15034957,
    "Japan": 0.07930284,
    "US": 0.049464009
}

risk_factor_encoding = {
    "Low": 0.50960313,
    "Medium": 0.34039992,
    "High": 0.14999694
}


model = load_model()

@st.cache_resource
def load_graph(graph_path):
    """Function to read the HTML graph content."""
    with open(graph_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    

    st.set_page_config(page_title='Thyroid Cancer ', layout='wide')
    
    # Custom CSS for styling
    st.markdown("""
        <style>
            body {
                background-color: white;
                color: #333;
            }
            .main-title {
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                color: #007BFF;
            }
            .subheader {
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-title">Data Analysis on Thyroid Cancer and Prediction Modeling</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3= st.tabs(["📃 Overview", "📊 Summary Statistics", "🤖 Prediction Model"])

    with tab1:
        # Custom CSS for styling the section with textbox-like borders
        st.markdown("""
            <style>
                .section-paragraph {
                    border: 2px solid #007BFF; /* Blue border */
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    background-color: #ffffff; /* White background, like a textbox */
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Soft shadow for a textbox effect */
                    font-family: 'Arial', sans-serif; /* Matching the look of textboxes */
                }
                .section-header {
                    color: #007BFF; /* Blue for the headers */
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .section-content {
                    color: #333; /* Dark text for readability */
                    font-size: 16px;
                    line-height: 1.6;
                }
                .highlight-text {
                    color: #D32F2F; /* Red color for important highlights */
                    font-weight: bold;
                }
            </style>
        """, unsafe_allow_html=True)

        # First section: Thyroid Cancer Prevention
        st.markdown('<div class="section-paragraph">', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Thyroid Cancer Prevention</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-content">Thyroid cancer prevention is crucial as it is one of the most common types of cancer.'
                    'Early detection and preventive measures can significantly improve outcomes, reducing both the morbidity and mortality '
                    'associated with the disease. Identifying risk factors such as family history, risk factor, and environmental influences '
                    'can lead to more effective screening strategies and personalized prevention plans.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-content">The data provided through this project plays a vital role in understanding the patterns and trends of thyroid cancer. '
                    'By analyzing large diagnostic datasets, we can identify correlations and risk factors that may not be immediately obvious, enabling early intervention for those at higher risk. '
                    'The ability to offer a free, user-friendly diagnostic tool based on this data empowers individuals to assess their own risk and take proactive steps towards prevention.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-content">In addition, the statistical insights and summaries derived from the dataset provide a broader understanding of thyroid cancer, '
                    'helping to guide healthcare policies, public awareness campaigns, and clinical practices aimed at reducing its impact. Ultimately, '
                    'this data-driven approach to thyroid cancer prevention provides both individual and collective benefits, fostering early detection, '
                    'improving health outcomes, and contributing to a more informed and health-conscious society.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Second section: Project Overview
        st.markdown('<div class="section-paragraph">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Project Overview</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-content">This project involves creating a comprehensive platform that leverages a large dataset to provide detailed summaries and statistics, '
                    'offering valuable insights to and explainations. The platform will feature a free tool that allows users to input responses to a questionnaire, generating a preliminary AI-predicted diagnostic based on their answers.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-content">Using a dense multilayered neural network, we can confidently predict a likely preliminary diagnostic to users from just a short questionaire. '
                    'We trained this model on a dataset of over 200,000 real diagnostics entry data, giving us a high correlation score. We also ran statistical analysis to determine significant findings using a more'
                    'traditional approach. We then considered our original model and our statistics to create a more effective final model, reducing overfitting and making diagnosis easier for the user. </p>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<p class="subheader">This section allows exploration of significant findings using traditional statistics. Each graph represents a different factor that our original data considered in the report.</p>', unsafe_allow_html=True)

        # Dropdown menu to select the graph
        graph_selection = st.selectbox(
            "Select a variable to explore:",
            ["Select a graph", "Age Distribution", "Gender Distribution", "By Country",
             "Diabetes Status", "Ethnicity Distribution", "Family History", "Iodine Deficiency",
             "Tumor Size", "Obesity", "Smoking Status", "Radiation Exposure", "TSH Levels", 
             "T3 Levels", "T4 Levels"
             ]
        )

        # Define graph paths mapping
        graph_paths = {
            "Age Distribution": "C:/TAMUCode/python/build4good2025/graphs/interactive_age_distribution.html",
            "Gender Distribution": "C:/TAMUCode/python/build4good2025/graphs/interactive_interaction_Gender_diagnosis_T3_Level.html",
            "By Country": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Country_diagnosis.html",
            "Diabetes Status": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Diabetes.html",
            "Ethnicity Distribution": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Ethnicity_diagnosis.html",
            "Family History": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Family_History_diagnosis.html",
            "Iodine Deficiency": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Iodine_Deficiency_diagnosis.html",
            "Tumor Size": "C:/TAMUCode/python/build4good2025/graphs/interactive_boxplot_Nodule_Size_by_diagnosis.html",
            "Obesity": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Obesity.html",
            "Smoking Status": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Smoking.html",
            "Radiation Exposure": "C:/TAMUCode/python/build4good2025/graphs/interactive_stacked_bar_Radiation_Exposure.html",
            "TSH Levels": "C:/TAMUCode/python/build4good2025/graphs/interactive_boxplot_TSH_Level_by_diagnosis.html",
            "T3 Levels": "C:/TAMUCode/python/build4good2025/graphs/interactive_boxplot_T3_Level_by_diagnosis.html",
            "T4 Levels": "C:/TAMUCode/python/build4good2025/graphs/interactive_boxplot_T4_Level_by_diagnosis.html"
        }

        graph_descriptions = {
            "Age Distribution": "The age distribution graph provides a visual representation of the age groups affected by thyroid cancer. "
                                "From the data, we concluded that there was a higher occurrence in individuals aged 40-60, with lower incidences "
                                "in younger and older age groups.",
            "Gender Distribution": "The gender distribution graph shows a higher incidence of thyroid cancer in females compared to males. "
                                   "This is consistent with known trends in thyroid cancer epidemiology.",
            "By Country": "The graph displays the distribution of thyroid cancer cases across different countries. This can highlight regions "
                          "with higher or lower rates of the disease, potentially offering insights into environmental or lifestyle factors.",
            "Diabetes Status": "This graph compares the rates of thyroid cancer between individuals with and without diabetes, offering insights "
                               "into potential correlations between the two conditions.",
            "Ethnicity Distribution": "This graph shows the distribution of thyroid cancer cases across different ethnic groups, helping identify "
                                      "whether certain groups may be more or less susceptible.",
            "Family History": "The graph visualizes the relationship between family history and the likelihood of developing thyroid cancer. "
                              "Family history is often considered a risk factor in many cancers.",
            "Iodine Deficiency": "This graph highlights the impact of iodine deficiency on thyroid cancer risk. Iodine deficiency is known to "
                                 "affect thyroid health, particularly in regions with low iodine intake.",
            "Tumor Size": "The tumor size distribution graph compares tumor sizes across different diagnoses, potentially revealing trends in "
                          "tumor growth associated with different stages of thyroid cancer.",
            "Obesity": "This graph compares the rates of thyroid cancer in individuals with obesity versus those without. Obesity is often "
                       "associated with increased cancer risk in many studies.",
            "Smoking Status": "The graph shows the correlation between smoking status and thyroid cancer incidence. Smoking is a known risk factor "
                              "for many cancers, and its impact on thyroid cancer is explored here.",
            "Radiation Exposure": "Radiation exposure is a known risk factor for many types of cancer, including thyroid cancer. This graph "
                                  "shows the correlation between radiation exposure and thyroid cancer occurrence.",
            "TSH Levels": "This graph visualizes the correlation between thyroid stimulating hormone (TSH) levels and thyroid cancer diagnosis. "
                          "TSH is a key regulator of thyroid function and can be an important marker in cancer diagnosis.",
            "T3 Levels": "The graph compares T3 hormone levels across different diagnoses, potentially revealing trends in thyroid function "
                          "associated with thyroid cancer.",
            "T4 Levels": "This graph compares T4 hormone levels across different diagnoses, potentially revealing trends in thyroid function "
                          "associated with thyroid cancer."
        }  

        # Check if a valid selection is made
        if graph_selection != "Select a graph":
            graph_path = graph_paths[graph_selection]

            html_content = load_graph(graph_path)

            components.html(html_content, height=600)

            st.markdown(f'<p class="section-content">{graph_descriptions.get(graph_selection, "No description available.")}</p>', unsafe_allow_html=True)


        else:
            st.write("Please select a graph to view.")
        

    
    with tab3:
        st.markdown('<p class="subheader">Prediction Model</p>', unsafe_allow_html=True)
        st.write("Fill out a short questionaire to determine your diagnostic.")

        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        tsh = st.number_input("TSH Level", value=1.5)
        t3 = st.number_input("T3 Level", value=2.0)
        t4 = st.number_input("T4 Level", value=1.0)
        nodule_size = st.number_input("Nodule Size", value=0.5)

        # Encoded inputs
        country = st.selectbox("Country", options=list(country_encoding.keys()))
        risk_factor = st.selectbox("Risk Factor", options=list(risk_factor_encoding.keys()))

        # Convert inputs to a numpy array
        input_data = np.array([[age, tsh, t3, t4, nodule_size, country_encoding[country], risk_factor_encoding[risk_factor]]])

        # Make prediction
        if st.button("Predict"):
            prediction = model.predict(input_data)
            st.write(f"Predicted Risk Score: {prediction[0][0]:.4f}")
            if prediction[0][0] > 0.5:
                st.write("High Risk of Thyroid Cancer")
            else:
                st.write("Low Risk of Thyroid Cancer")
        st.markdown('<p class="section-content">This model is based on a dense multilayered neural network trained on a dataset of over 200,000 real diagnostics entry data. '
                    'It provides a preliminary AI-predicted diagnostic based on the questionnaire responses.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
