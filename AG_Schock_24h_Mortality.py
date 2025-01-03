
import streamlit as st     
import subprocess
import sys


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import dice_ml
import shap
import zipfile
import re
from PyPDF2 import PdfReader
import io
import re
from collections import defaultdict




st.title("AG Schock Mortality Modell")


st.markdown(
    """
    The model has been pre-trained on data from cardiogenic shock patients, specifically those classified as SCAI stage C or higher.
    
    ### Inclusion criteria:

    - Lactate levels greater than 2 mmol/L
    - Administration of vasopressors
    - A cardiogenic origin of the shock

    ### Interested in learning more about our models?

    - Explore our research papers for detailed insights and findings.

"""
)

uploaded_file = st.file_uploader("Please upload a ZIP file with models", type="zip")

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the uploaded ZIP file as a byte stream            
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            # Load each model directly from the zip file into memory
            # Load s_models
            with zip_ref.open('s_model_24h.pkl') as file:
                s_model_24h = pickle.load(file)

            with zip_ref.open('s_model_2d.pkl') as file:
                s_model_2d = pickle.load(file)

            with zip_ref.open('s_model_3d.pkl') as file:
                s_model_3d = pickle.load(file)

            with zip_ref.open('s_model_4d.pkl') as file:
                s_model_4d = pickle.load(file)

            with zip_ref.open('s_model_5d.pkl') as file:
                s_model_5d = pickle.load(file)

            with zip_ref.open('s_model_6d.pkl') as file:
                s_model_6d = pickle.load(file)

            with zip_ref.open('s_model_1w.pkl') as file:
                s_model_1w = pickle.load(file)

            with zip_ref.open('s_model_8d.pkl') as file:
                s_model_8d = pickle.load(file)

            with zip_ref.open('s_model_9d.pkl') as file:
                s_model_9d = pickle.load(file)

            with zip_ref.open('s_model_10d.pkl') as file:
                s_model_10d = pickle.load(file)

            with zip_ref.open('s_model_11d.pkl') as file:
                s_model_11d = pickle.load(file)

            with zip_ref.open('s_model_12d.pkl') as file:
                s_model_12d = pickle.load(file)

            with zip_ref.open('s_model_13d.pkl') as file:
                s_model_13d = pickle.load(file)

            with zip_ref.open('s_model_2w.pkl') as file:
                s_model_2w = pickle.load(file)

            with zip_ref.open('s_model_15d.pkl') as file:
                s_model_15d = pickle.load(file)

            with zip_ref.open('s_model_3w.pkl') as file:
                s_model_3w = pickle.load(file)

            # Load p_models
            with zip_ref.open('p_model_24h.pkl') as file:
                p_model_24h = pickle.load(file)

            with zip_ref.open('p_model_2d.pkl') as file:
                p_model_2d = pickle.load(file)

            with zip_ref.open('p_model_3d.pkl') as file:
                p_model_3d = pickle.load(file)

            with zip_ref.open('p_model_4d.pkl') as file:
                p_model_4d = pickle.load(file)

            with zip_ref.open('p_model_5d.pkl') as file:
                p_model_5d = pickle.load(file)

            with zip_ref.open('p_model_6d.pkl') as file:
                p_model_6d = pickle.load(file)

            with zip_ref.open('p_model_1w.pkl') as file:
                p_model_1w = pickle.load(file)

            with zip_ref.open('p_model_8d.pkl') as file:
                p_model_8d = pickle.load(file)

            with zip_ref.open('p_model_9d.pkl') as file:
                p_model_9d = pickle.load(file)

            with zip_ref.open('p_model_10d.pkl') as file:
                p_model_10d = pickle.load(file)

            with zip_ref.open('p_model_11d.pkl') as file:
                p_model_11d = pickle.load(file)

            with zip_ref.open('p_model_12d.pkl') as file:
                p_model_12d = pickle.load(file)

            with zip_ref.open('p_model_13d.pkl') as file:
                p_model_13d = pickle.load(file)

            with zip_ref.open('p_model_2w.pkl') as file:
                p_model_2w = pickle.load(file)

            with zip_ref.open('p_model_15d.pkl') as file:
                p_model_15d = pickle.load(file)

            with zip_ref.open('p_model_3w.pkl') as file:
                p_model_3w = pickle.load(file)

            # Load scalers
            with zip_ref.open('scaler_24h.pkl') as file:
                scaler_24h = pickle.load(file)

            with zip_ref.open('scaler_2d.pkl') as file:
                scaler_2d = pickle.load(file)

            with zip_ref.open('scaler_3d.pkl') as file:
                scaler_3d = pickle.load(file)

            with zip_ref.open('scaler_4d.pkl') as file:
                scaler_4d = pickle.load(file)

            with zip_ref.open('scaler_5d.pkl') as file:
                scaler_5d = pickle.load(file)

            with zip_ref.open('scaler_6d.pkl') as file:
                scaler_6d = pickle.load(file)

            with zip_ref.open('scaler_1w.pkl') as file:
                scaler_1w = pickle.load(file)

            with zip_ref.open('scaler_8d.pkl') as file:
                scaler_8d = pickle.load(file)

            with zip_ref.open('scaler_9d.pkl') as file:
                scaler_9d = pickle.load(file)

            with zip_ref.open('scaler_10d.pkl') as file:
                scaler_10d = pickle.load(file)

            with zip_ref.open('scaler_11d.pkl') as file:
                scaler_11d = pickle.load(file)

            with zip_ref.open('scaler_12d.pkl') as file:
                scaler_12d = pickle.load(file)

            with zip_ref.open('scaler_13d.pkl') as file:
                scaler_13d = pickle.load(file)

            with zip_ref.open('scaler_2w.pkl') as file:
                scaler_2w = pickle.load(file)

            with zip_ref.open('scaler_15d.pkl') as file:
                scaler_15d = pickle.load(file)

            with zip_ref.open('scaler_3w.pkl') as file:
                scaler_3w = pickle.load(file)
            
            with zip_ref.open('exp_dice.pkl') as file:
                exp_dice = pickle.load(file)    
                        
            with zip_ref.open('optimal_thresholds.pkl') as file:
                optimal_thresholds = pickle.load(file)

    
        st.success("Models loaded successfully!")
    
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except pickle.UnpicklingError:
        st.error("Error unpickling one of the files. Make sure they are valid pickle files.")
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid zip file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
                
    class model_24h_mortality:
        def __init__(self, 
                    s_model_24h=s_model_24h, p_model_24h=p_model_24h, scaler_24h=scaler_24h,
                    s_model_2d=s_model_2d, p_model_2d=p_model_2d, scaler_2d=scaler_2d,
                    s_model_3d=s_model_3d, p_model_3d=p_model_3d, scaler_3d=scaler_3d,
                    s_model_4d=s_model_4d, p_model_4d=p_model_4d, scaler_4d=scaler_4d,
                    s_model_5d=s_model_5d, p_model_5d=p_model_5d, scaler_5d=scaler_5d,
                    s_model_6d=s_model_6d, p_model_6d=p_model_6d, scaler_6d=scaler_6d,
                    s_model_1w=s_model_1w, p_model_1w=p_model_1w, scaler_1w=scaler_1w,
                    s_model_8d=s_model_8d, p_model_8d=p_model_8d, scaler_8d=scaler_8d,
                    s_model_9d=s_model_9d, p_model_9d=p_model_9d, scaler_9d=scaler_9d,
                    s_model_10d=s_model_10d, p_model_10d=p_model_10d, scaler_10d=scaler_10d,
                    s_model_11d=s_model_11d, p_model_11d=p_model_11d, scaler_11d=scaler_11d,
                    s_model_12d=s_model_12d, p_model_12d=p_model_12d, scaler_12d=scaler_12d,
                    s_model_13d=s_model_13d, p_model_13d=p_model_13d, scaler_13d=scaler_13d,
                    s_model_2w=s_model_2w, p_model_2w=p_model_2w, scaler_2w=scaler_2w,
                    s_model_15d=s_model_15d, p_model_15d=p_model_15d, scaler_15d=scaler_15d,
                    s_model_3w=s_model_3w, p_model_3w=p_model_3w, scaler_3w=scaler_3w,
                    optimal_thresholds=optimal_thresholds, exp_dice=exp_dice):

            # 24h Modelle
            self.s_model_24h = s_model_24h
            self.p_model_24h = p_model_24h
            self.scaler_24h = scaler_24h

            # 2D Modelle
            self.s_model_2d = s_model_2d
            self.p_model_2d = p_model_2d
            self.scaler_2d = scaler_2d

            # 3D Modelle
            self.s_model_3d = s_model_3d
            self.p_model_3d = p_model_3d
            self.scaler_3d = scaler_3d

            # 4D Modelle
            self.s_model_4d = s_model_4d
            self.p_model_4d = p_model_4d
            self.scaler_4d = scaler_4d

            # 5D Modelle
            self.s_model_5d = s_model_5d
            self.p_model_5d = p_model_5d
            self.scaler_5d = scaler_5d

            # 6D Modelle
            self.s_model_6d = s_model_6d
            self.p_model_6d = p_model_6d
            self.scaler_6d = scaler_6d

            # 1W Modelle
            self.s_model_1w = s_model_1w
            self.p_model_1w = p_model_1w
            self.scaler_1w = scaler_1w

            # 8D Modelle
            self.s_model_8d = s_model_8d
            self.p_model_8d = p_model_8d
            self.scaler_8d = scaler_8d

            # 9D Modelle
            self.s_model_9d = s_model_9d
            self.p_model_9d = p_model_9d
            self.scaler_9d = scaler_9d

            # 10D Modelle
            self.s_model_10d = s_model_10d
            self.p_model_10d = p_model_10d
            self.scaler_10d = scaler_10d

            # 11D Modelle
            self.s_model_11d = s_model_11d
            self.p_model_11d = p_model_11d
            self.scaler_11d = scaler_11d

            # 12D Modelle
            self.s_model_12d = s_model_12d
            self.p_model_12d = p_model_12d
            self.scaler_12d = scaler_12d

            # 13D Modelle
            self.s_model_13d = s_model_13d
            self.p_model_13d = p_model_13d
            self.scaler_13d = scaler_13d

            # 2W Modelle
            self.s_model_2w = s_model_2w
            self.p_model_2w = p_model_2w
            self.scaler_2w = scaler_2w

            # 15D Modelle
            self.s_model_15d = s_model_15d
            self.p_model_15d = p_model_15d
            self.scaler_15d = scaler_15d

            # 3W Modelle
            self.s_model_3w = s_model_3w
            self.p_model_3w = p_model_3w
            self.scaler_3w = scaler_3w

            # Experiment-Dice Modell
            self.optimal_thresholds = optimal_thresholds
            self.exp_dice = exp_dice
            



        def predict(self, X_test_full):
            
            
            st.write('### 24hours-Mortality Prediction')



            df = self.scaler.transform(X_test_full)
            df = pd.DataFrame(df, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                                            'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20',
                                            'PC21', 'PC22'])

            y_pred_t_model = self.t_model.predict(df)
            y_pred_t_model = pd.Series(y_pred_t_model)
            y_pred_t_model.index = X_test_full.index
            
            #print(y_pred_t_model)
            
            
            for index, pred in enumerate(y_pred_t_model):
                    
                if pred == 0:
                    #y_pred = self.cohort1_model.predict(X_test_full)


                    
                    y_pred_proba = self.cohort1_model.predict_proba(X_test_full)[:, 1]
                    optimal_threshold = 0.5493
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                    st.write("The 24h-Mortality ist predicted with 0.91 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next day:", y_pred[0])
                    #st.write("Mortality probability  for the next day:", y_pred_proba[0])


                if pred == 1:
                    #y_pred = self.cohort2_model.predict(X_test_full)


                    
                    y_pred_proba = self.cohort2_model.predict_proba(X_test_full)[:, 1]
                    optimal_threshold = 0.5562
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                    st.write("The 24h-Mortality ist predicted with 0.72 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next day:", y_pred[0])
                    #st.write("Mortality probability  for the next day:", y_pred_proba[0])
                    
                    
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
                    
            st.write('### 3days-Mortality Prediction')
                    
            df = self.scaler_3d.transform(X_test_full)
            df = pd.DataFrame(df, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                                            'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20',
                                            'PC21', 'PC22'])

            y_pred_t_model = self.t_model_3d.predict(df)
            y_pred_t_model = pd.Series(y_pred_t_model)
            y_pred_t_model.index = X_test_full.index
            
            #print(y_pred_t_model)
            
            
            for index, pred in enumerate(y_pred_t_model):
                    
                if pred == 0:
                    #y_pred = self.cohort1_model_3d.predict(X_test_full)



                    y_pred_proba = self.cohort1_model_3d.predict_proba(X_test_full)[:, 1]
                    optimal_threshold = 0.4048
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                    st.write("The 3d-Mortality ist predicted with 0.81 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next 3 days:", y_pred[0])
                    #st.write("Mortality probability  for the next 3 days:", y_pred_proba[0])

                if pred == 1:
                    #y_pred = self.cohort2_model_3d.predict(X_test_full)


                    y_pred_proba = self.cohort2_model_3d.predict_proba(X_test_full)[:, 1]
                    optimal_threshold = 0.8712
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

                    st.write("The 3d-Mortality ist predicted with 0.34 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next 3 days:", y_pred[0])
                    #st.write("Mortality probability  for the next 3 days:", y_pred_proba[0])


            return y_pred, y_pred_proba
        
        
        def predict(self, df_input):
            """
            Führt Vorhersagen für jeden Tag mit den entsprechenden Modellen durch und bestimmt,
            ob die Vorhersage in die gut oder schlecht vorhersagbare Kohorte fällt.

            Args:
                df_input (DataFrame): Eingabedaten mit den gleichen Features wie beim Training.
                optimal_thresholds (dict): Dictionary mit optimalen Schwellenwerten pro Tag.
                                        Schlüssel sind die Tagesnummern, Werte sind die Schwellenwerte.

            Returns:
                DataFrame: Tabelle mit Vorhersagen, Wahrscheinlichkeiten und Kohorteninformationen.
            """
            
            results = []

            # Iteriere über alle Tage, für die ein Modell existiert
            for day in range(1, 16):
                model_key = f"final_model_{day}d"  # Name des Modells
                scaler_key = f"scaler_{day}d"      # Name des Scalers
                outlier_model_key = f"p_model_{day}d"  # Outlier-Erkennungsmodell

                # Überprüfe, ob das Modell für diesen Tag existiert
                if not hasattr(self, model_key):
                    st.warning(f"Kein Modell für Tag {day} gefunden.")
                    continue

                # Extrahiere den optimalen Schwellenwert für den aktuellen Tag
                optimal_threshold = self.optimal_thresholds.get(day, 0.5)  # Standardwert 0.5, falls nicht vorhanden

                # Hole das Tagesmodell und den zugehörigen Scaler
                final_model = getattr(self, model_key)
                scaler = getattr(self, scaler_key)
                outlier_model = getattr(self, outlier_model_key)

                # Skaliere die Eingabedaten
                df_scaled = scaler.transform(df_input)

                # Prädiziere die Outlier-Kohorte
                outliers = outlier_model.predict(df_scaled)
                cohort = 'Gute Kohorte' if outliers[0] == 0 else 'Schlechte Kohorte'

                # Prädiziere die Wahrscheinlichkeit der positiven Klasse
                pred_proba = final_model.predict_proba(df_scaled)[:, 1]
                prediction = (pred_proba >= optimal_threshold).astype(int)

                # Speichere das Ergebnis
                results.append({
                    "Day": f"Day {day}",
                    "Mortality_Probability": pred_proba[0],
                    "Prediction": prediction[0],
                    "subgroup": cohort
                })

            # Ergebnisse als DataFrame darstellen
            results_df = pd.DataFrame(results)
            
            st.write("## Prädiktionsergebnisse für jeden Tag")
            st.table(results_df)

            return results_df


        
        def generate_counterfactuals(self, X_test, total_CFs=5, desired_class=0, features_to_vary=None, permitted_range=None):
            if features_to_vary is None:
                features_to_vary = []
            if permitted_range is None:
                permitted_range = {}

            df = self.scaler.transform(X_test)
            df = pd.DataFrame(df, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                                        'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20',
                                        'PC21', 'PC22'])

            y_pred_t_model = self.t_model.predict(df)
            y_pred_t_model = pd.Series(y_pred_t_model)
            y_pred_t_model.index = X_test.index

            for index, pred in enumerate(y_pred_t_model):
                if pred == 0:
                    #print("The Patient belongs to cluster 1")
                    e1 = self.exp1.generate_counterfactuals(X_test, total_CFs=total_CFs, desired_class=desired_class, features_to_vary=features_to_vary, permitted_range=permitted_range)
                    e1.visualize_as_dataframe(show_only_changes=True)
                    result_df = e1.cf_examples_list[0].final_cfs_df
                    result_df = pd.DataFrame(result_df)

                if pred == 1:
                    #print("The Patient belongs to cluster 2")
                    e1 = self.exp2.generate_counterfactuals(X_test, total_CFs=total_CFs, desired_class=desired_class, features_to_vary=features_to_vary, permitted_range=permitted_range)
                    e1.visualize_as_dataframe(show_only_changes=True)
                    result_df = e1.cf_examples_list[0].final_cfs_df
                    result_df = pd.DataFrame(result_df)

            return result_df
        
        

        # Function to create a SHAP waterfall plot
        def plot_shap_waterfall(self, X_test):
            
            st.write('### SHAP-explanations for 24hours_mortality prediction')

            # Preprocess the test data
            df = self.scaler.transform(X_test)
            df = pd.DataFrame(df, columns=[
                'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20',
                'PC21', 'PC22'
            ])

            # Predict using the model
            y_pred_t_model = self.t_model.predict(df)
            y_pred_t_model = pd.Series(y_pred_t_model)
            y_pred_t_model.index = X_test.index

            # Create SHAP explainer and values
            for index, pred in enumerate(y_pred_t_model):
                if pred == 0:
                    explainer = shap.TreeExplainer(self.cohort1_model)
                else:  # Assuming it's binary classification, hence the `else` for pred == 1
                    explainer = shap.TreeExplainer(self.cohort2_model)

                # Compute SHAP values
                shap_values = explainer(X_test.iloc[[index]])  # Obtain an Explanation object

                # Plot SHAP waterfall plot
                shap.initjs()
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(shap_values[0], max_display=10)  # Display only top features

                # Save plot to an in-memory buffer instead of disk
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)

                # Display the image in Streamlit directly from memory
                st.image(buffer)

                # Close the plot to avoid memory leaks
                plt.close()
                
                
                #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
                
            st.write('### SHAP-explanations for 3days_mortality prediction')
                
            # Preprocess the test data
            df = self.scaler.transform(X_test)
            df = pd.DataFrame(df, columns=[
                'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20',
                'PC21', 'PC22'
            ])

            # Predict using the model
            y_pred_t_model = self.t_model_3d.predict(df)
            y_pred_t_model = pd.Series(y_pred_t_model)
            y_pred_t_model.index = X_test.index

            # Create SHAP explainer and values
            for index, pred in enumerate(y_pred_t_model):
                if pred == 0:
                    explainer = shap.TreeExplainer(self.cohort1_model_3d)
                else:  # Assuming it's binary classification, hence the `else` for pred == 1
                    explainer = shap.TreeExplainer(self.cohort2_model_3d)

                # Compute SHAP values
                shap_values = explainer(X_test.iloc[[index]])  # Obtain an Explanation object

                # Plot SHAP waterfall plot
                shap.initjs()
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(shap_values[0], max_display=10)  # Display only top features

                # Save plot to an in-memory buffer instead of disk
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)

                # Display the image in Streamlit directly from memory
                st.image(buffer)

                # Close the plot to avoid memory leaks
                plt.close()
                
                
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

            # Display SHAP explanation
            st.write("""
            ### How to Read the SHAP Waterfall Plot:
            
            - **Baseline**: The baseline represents the average predicted value.
            - **Positive Values (Red)**: Features that increase the predicted value (towards mortality).
            - **Negative Values (Blue)**: Features that decrease the predicted value (towards survival).
            - **Arrows**: Indicate how much each feature contributes to the prediction.
            
            The waterfall plot helps you understand which features have the most significant impact on the prediction.
            """)
                
                
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-





    # Modell initialisieren
    model = model_24h_mortality(t_model, cohort1_model, scaler, cohort2_model, exp1, exp2)
    st.session_state['model'] = model
    
    st.session_state['data_formatted'] = False

    User_Input = st.radio("How do you prefer to input the data?", ("Automatic input form pdf/excel", "Manual input"))
    if User_Input == "Automatic input form pdf/excel":        
        
        # Synonym-Mapping für die Features in Deutsch und Englisch
        synonyms = {
            'AgeOnInclusion': ['Age', 'Alter', 'Years', 'Years of Age', 'Lebensalter'],
            'RRSysWeightedMeanValue': ['systolischer arterieller Blutdruck', 'systolischer Blutdruck', 'systolic blood pressure', 'systolic BP', 'blood pressure systolic', 'systolische Blutdruckmessung', 'RR systolisch', 'RRsys'],
            'RRDiaWeightedMeanValue': ['diastolischer arterieller Blutdruck', 'diastolischer Blutdruck', 'diastolic blood pressure', 'diastolic BP', 'blood pressure diastolic', 'diastolische Blutdruckmessung', 'RR diastolisch', 'RRdia'],
            'sO2WeightedMeanValue': ['sauerstoffsättigung', 'oxygen saturation', 'SpO2', 'sO2', 'oxygen level', 'O2 sat', 'Sauerstofflevel', 'O2-Sättigung'],
            'TempWeightedMean': ['Körpertem peratur', 'körpertemperatur', 'body temperature', 'temp', 'temperature', 'core temperature', 'Körperwärme', 'Körpertemp'],
            'PHWeightedMean': ['ph', 'blood pH', 'pH level', 'pH-Wert', 'Säure-Basen-Status', 'Blut-pH'],
            'LactateWeightedMean': ['Lakt', 'laktatspiegel', 'lactate level', 'blood lactate', 'Laktat', 'Lactate', 'Serumlaktat', 'Laktatkonzentration'],
            'gluWeightedMean': ['Glu','blutzucker', 'blood glucose', 'glucose', 'Blutglukose', 'Blutzuckerspiegel', 'blood sugar', 'Glucoselevel', 'Glukosespiegel'],
            'HCO3WeightedMean': ['bicarbonat', 'bicarbonate', 'HCO3', 'Bikarbonatspiegel', 'Hydrogencarbonate', 'Serum-Bicarbonat', 'HCO₃⁻'],
            'DayNumber': ['Day number', 'Tag auf ITS', 'Hospital Day', 'Stay Day', 'Hospital Day Number', 'Verweildauer', 'Tage seit Aufnahme', 'ITS-Tag'],
            'PO2WeightedMean': ['sauerstoffpartialdruck', 'partial pressure of oxygen', 'PaO2', 'pO2', 'Oxygen partial pressure', 'Oxygen Tension', 'arterieller Sauerstoffdruck', 'Sauerstoffdruck'],
            'PCO2WeightedMean': ['kohlendioxidpartialdruck', 'partial pressure of carbon dioxide', 'PaCO2', 'pCO2', 'Carbon dioxide partial pressure', 'arterieller Kohlendioxid-Druck', 'CO2-Druck', 'CO₂-Partialdruck'],
            'HBWeightedMean': ['hämoglobin', 'hemoglobin', 'Hb', 'blood hemoglobin', 'Hämoglobinspiegel', 'Hb-Wert', 'Bluthämoglobin'],
            'leucoWeightedMean': ['leukozyten', 'leukocytes', 'white blood cells', 'WBC', 'Leukocytenzahl', 'Weiße Blutkörperchen', 'Leukos'],
            'ureaWeightedMean': ['harnstoff', 'urea', 'blood urea', 'Urea nitrogen', 'Harnstoffstickstoff', 'BUN', 'Harnstoffkonzentration'],
            'HRWeightedMean': ['herzfrequenz', 'heart rate', 'pulse', 'HR', 'Pulsfrequenz', 'Puls', 'Cardiac rate', 'Herzschlagfrequenz'],
            'NaWeightedMean': ['Natrium(POC)','Natrium', 'sodium', 'Na', 'Serumnatrium', 'Sodium level', 'Natriumspiegel', 'Natriumwert'],
            'KWeightedMean': ['kalium', 'potassium', 'K', 'Serumkalium', 'Potassium level', 'Kaliumspiegel', 'Kaliumwert'],
            'ClWeightedMean': ['chlorid', 'chloride', 'Cl', 'Serumchlorid', 'Chloridspiegel', 'Chloridion', 'Chloridwert'],
            'Height': ['Height', 'Größe', 'Body height', 'Körpergröße', 'Stature', 'Länge', 'Körperlänge'],
            'Weight': ['Weight', 'Gewicht', 'Body weight', 'Körpergewicht', 'Mass', 'Masse'],
            'plateletsWeightedMean': ['thrombozyten', 'platelets', 'Thrombocytes', 'Plättchen', 'Platelet count', 'Thrombozytenzahl', 'Blutplättchen']
        }





        # Funktion zur Verarbeitung der hochgeladenen PDF-Datei
        def process_pdf(file):
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            return text






        def calculate_weighted_means(df):
            # Überprüfen, ob die 'Time'-Zeile vorhanden ist
            if 'Time' not in df.index:
                raise KeyError("'Time'-row is not in the DataFrame.")
        
            # Zeitstempel extrahieren
            times = df.loc['Time'].values
            # Entfernen der 'Time'-Zeile aus dem DataFrame
            df = df.drop(index='Time')
            
            weighted_means = {}
        
            # Durch alle Features (Zeilen) iterieren
            for feature in df.index:
                # Datenbereinigung und Umwandlung in numerische Werte
                values = pd.to_numeric(df.loc[feature], errors='coerce')  # Umwandlung und Ersetzung von Fehlern durch NaN
        
                # Zeitstempel ohne NaN-Werte
                times_non_nan = times[~values.isna()]
                values_non_nan = values.dropna()  # Entferne NaN-Werte
                
                if len(values_non_nan) > 0 and len(times_non_nan) > 0 and len(values_non_nan) == len(times_non_nan):
                    # Berechnung des gewichteten Mittelwerts, ignoriert NaN-Werte
                    weighted_mean = np.mean(values_non_nan)  # Berechnung des Mittelwerts ohne NaNs
                    weighted_means[feature] = weighted_mean
                else:
                    weighted_means[feature] = np.nan  # Wenn es nicht genug Daten gibt, NaN zurückgeben
        
            # In eine Tabelle umwandeln (DataFrame)
            result_df = pd.DataFrame(weighted_means, index=['WeightedMean']).transpose()
            
            return result_df
        

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-





        def process_excel(file, synonyms):
            # Lesen der Excel-Datei
            df = pd.read_excel(file)
            df = df.dropna(how='all')  # Entfernt leere Zeilen
            df = df.dropna(axis=1, how='all')  # Entfernt leere Spalten

            # Muster für Zeitangaben im Format HH:MM:SS oder HH:MM
            time_pattern = r'\b\d{2}:\d{2}(?::\d{2})?\b'
            # Muster für Datum-Uhrzeit-Kombinationen im Format YYYY-MM-DD HH:MM:SS oder YYYY-MM-DD HH:MM
            datetime_pattern = r'\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?\b'

            # Datenstruktur für extrahierte Daten
            data = {'Time': []}
            time_columns = []  # Liste zur Speicherung der Spalten mit Zeitangaben

            # Gehe durch die Spalten des DataFrames und finde die Spalten mit Zeitangaben
            for col in df.columns:
                for cell_value in df[col]:
                    if re.search(f'({time_pattern})|({datetime_pattern})', str(cell_value)):
                        time_columns.append(col)
                        break  # Wenn eine Zeitangabe in der Spalte gefunden wurde, zur nächsten Spalte gehen

            # Durch die Zeilen und Spalten iterieren
            for index, row in df.iterrows():
                # Finde und konvertiere alle Zeitangaben in der Zeile in den Zeitspalten
                times = []
                for col in time_columns:
                    cell_value = str(row[col])
                    found_times = re.findall(f'({time_pattern})|({datetime_pattern})', cell_value)
                    found_times = [t for t in (item for sublist in found_times for item in sublist) if t]
                    found_times = [format_time(t) for t in found_times]
                    times.extend(found_times)

                # Wenn Zeitangaben gefunden wurden, füge sie zur Datenstruktur hinzu
                if times:
                    data['Time'].extend(times)

                # Gehe durch die Spalten der aktuellen Zeile, um Synonyme zu finden
                for feature, synonym_list in synonyms.items():
                    for synonym in synonym_list:
                        # Verwende \b um das Synonym als eigenständiges Wort zu suchen
                        pattern = rf'\b{re.escape(synonym)}\b'
                        if any(re.search(pattern, str(row[col]), re.IGNORECASE) for col in df.columns):
                            # Extrahiere Werte in der aktuellen Zeile aus den Zeitspalten
                            values = []
                            for col in time_columns:
                                cell_value = str(row[col]).strip()
                                if cell_value == '' or pd.isna(row[col]):  # Überprüfen auf leere Zellen oder NaN-Werte
                                    values.append(None)  # Füge `None` hinzu, um leere Zellen zu repräsentieren
                                else:
                                    cell_value = re.sub(r',', '.', cell_value)
                                    value_match = re.findall(r'(-?\d+\.?\d*)', cell_value)
                                    if value_match:  # Nur hinzufügen, wenn tatsächlich Werte gefunden wurden
                                        values.extend([float(val) for val in value_match])
                                    else:
                                        values.append(None)  # Wenn kein Wert gefunden wurde, `None` hinzufügen
                            # Füge die extrahierten Werte zum Feature hinzu
                            if values:
                                if feature not in data:
                                    data[feature] = []
                                data[feature].extend(values)
                            break  # Feature gefunden, keine weiteren Synonyme prüfen

            # Überprüfen, ob die Anzahl der Werte größer ist als die Anzahl der Zeitangaben
            for feature, values in data.items():
                if feature == 'Time':
                    continue
                if len(values) > len(data['Time']):
                    return f"Error: Die Anzahl der erkannten Werte für '{feature}' ist größer als die Anzahl der Zeitangaben. Bitte überprüfen Sie die Daten in dieser Variable."

            # Erstellen eines DataFrames, der die Zeitangaben den Variablen zuordnet
            final_data = {'Time': data['Time']}
            for feature, values in data.items():
                if feature == 'Time':
                    continue
                final_data[feature] = values + [None] * (len(final_data['Time']) - len(values))

            result_df = pd.DataFrame(final_data)

            # Verstecke leere Spalten, die keine Daten enthalten
            result_df = result_df.dropna(axis=1, how='all')

            return result_df

        def format_time(time_str):
            if len(time_str) == 5:  # Wenn Zeit im Format HH:MM vorliegt
                return time_str + ":00"
            return time_str  # Bereits im Format HH:MM:SS


        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


        def replace_synonyms(text, synonyms):

            for standard_term, synonym_list in synonyms.items():
                for synonym in synonym_list:
                    # Ersetze das Synonym nur, wenn es ein vollständiges Wort ist
                    pattern = rf'(?<!\w){re.escape(synonym)}(?!\w)'
                    text = re.sub(pattern, standard_term, text, flags=re.IGNORECASE)
                #print(text)
            return text

                





        def replace_time(text):
            """
            Erkenne unterschiedliche Zeitformate und normalisiere sie zu einem einheitlichen Format (HH:MM).
            Gibt den Text mit normalisierten Zeiten zurück.
            """
            time_pattern = r'(\d{1,2})[:.](\d{2})'
            
            # Funktion, die für jeden gefundenen Treffer die normalisierte Zeit zurückgibt
            def replace_with_normalized(match):
                hours, minutes = match.groups()
                normalized_time = f"{int(hours):02d}:{int(minutes):02d}"
                return normalized_time

            # Ersetze alle Zeitangaben im Text mit dem normalisierten Format
            normalized_text = re.sub(time_pattern, replace_with_normalized, text)
            
            return normalized_text


        # Hauptfunktion zur Textverarbeitung und Extraktion
        def text_extraction(input_text, synonyms):

            # Schritt 1: Entferne unerwünschte Zeichen wie ":" aus dem Text
            cleaned_text = re.sub(r'[:]', '', input_text)

            # Schritt 2: Normalisiere die Uhrzeiten
            cleaned_text = replace_time(cleaned_text)

            # Schritt 3: Ersetze die Synonyme im Text
            normalized_text = replace_synonyms(cleaned_text, synonyms)

            # Erstelle ein Muster, um die Uhrzeiten mit zugehörigen Daten zu finden
            time_pattern =  r'(\d{1,2})[:.](\d{2})'
            data_pattern = r'(\w+(?: \w+)?)(?:\s+(-?\d+(?:,\d+)?))'  # Muster für "Begriff Zahl"

            # Initialisiere das Daten-Dictionary mit Zeiten
            data_dict = {'Time': []}
            features = list(synonyms.keys())
            for feature in features:
                data_dict[feature] = []

            # Schritt 4: Iteriere über den Text und ordne die Werte den Uhrzeiten zu
            current_time = None
            for line in normalized_text.splitlines():
                # Prüfe, ob die Zeile eine Uhrzeit enthält
                time_match = re.findall(time_pattern, line)
                #print(time_match)
                if time_match:

                #matches = re.findall(time_pattern, text)
                    #normalized_times = []
                    for match in time_match:
                        hours, minutes = match
                        current_time = f"{int(hours):02d}:{int(minutes):02d}"
                        #normalized_times.append(normalized_time)

                    #print(current_time)

                    #current_time = time_match.group(1)  # Setze die aktuelle Uhrzeit
                    data_dict['Time'].append(current_time)
                    # Füge None für alle Features hinzu, um für die Uhrzeit Platz zu schaffen
                    for feature in features:
                        data_dict[feature].append(None)
                elif current_time:
                    # Wenn es keine Uhrzeit gibt, extrahiere die Begriffe und Werte
                    data = re.findall(data_pattern, line)
                    for term, value in data:
                        term = term.strip()
                        value = float(value.replace(',', '.'))  # Konvertiere zu float
                        if term in features:
                            # Aktualisiere den Wert für die letzte erkannte Uhrzeit
                            data_dict[term][-1] = value

            # Schritt 5: Erstelle den DataFrame
            df = pd.DataFrame(data_dict)


            def clean_dataset(df):
                # Schritt 1: Entferne Zeilen, die komplett aus NaN bestehen
                df_cleaned = df.dropna(axis=0, how='all')

                # Schritt 2: Entferne Spalten, die nur einen Wert und ansonsten NaN enthalten
                # Das gilt nur für die Spalten außer 'Time'
                #first_row = df_cleaned.iloc[0].values.flatten().tolist()
                #print(first_row)
                #df_cleaned = df_cleaned.drop(df_cleaned.index[2]).reset_index(drop=True)
                #df_cleaned = df_cleaned.drop(df_cleaned.index[0])
                #df_cleaned = df_cleaned.drop(df_cleaned.index[0])
                #non_time_columns = df_cleaned.columns.difference(['Time'])
                
                # Filter: Behalte nur die Spalten, die mindestens einen Nicht-NaN-Wert enthalten
                df_cleaned = df_cleaned.dropna(axis=1, how='all')


                non_time_rows = df_cleaned.index[df_cleaned.index != 'Time']

                # Lösche Spalten, die nur NaN-Werte in non_time_rows haben
                df_cleaned = df_cleaned.dropna(axis=1, how='all', subset=non_time_rows)


                return df_cleaned
            
            df = clean_dataset(df.T)

            return df.T

        #if st.button("Data input as PDF / Excel"):

        feature_names = {
                    'RRSysWeightedMeanValue': 'Systolic Blood Pressure (mmHg)',
                    'RRDiaWeightedMeanValue': 'Diastolic Blood Pressure (mmHg)',
                    'sO2WeightedMeanValue': 'Oxygen Saturation (%)',
                    'TempWeightedMean': 'Body Temperature (°C)',
                    'PHWeightedMean': 'Blood pH',
                    'LactateWeightedMean': 'Blood Lactate Level (mmol/L)',
                    'gluWeightedMean': 'Blood Glucose Level (mg/dL)',
                    'HCO3WeightedMean': 'Bicarbonate (HCO3) (mmol/L)',
                    'PO2WeightedMean': 'Partial Pressure of Oxygen (PaO2) (mmHg)',
                    'PCO2WeightedMean': 'Partial Pressure of Carbon Dioxide (PaCO2) (mmHg)',
                    'HBWeightedMean': 'Hemoglobin (HB) (g/dl)',
                    'leucoWeightedMean': 'Leukocytes (G/l)',
                    'ureaWeightedMean': 'Blood Urea (mg/dl)',
                    'HRWeightedMean': 'Heart Rate (BPM)',
                    'NaWeightedMean': 'Sodium (Na+) (mmol/L)',
                    'KWeightedMean': 'Potassium (K+) (mmol/L)',
                    'ClWeightedMean': 'Chloride (Cl-) (mmol/L)',
                    'plateletsWeightedMean': 'Platelets (G/l)'
        }

        # Frage, wie viele Dateien hochgeladen werden sollen
        num_files = st.number_input("How many files would you like to upload?", min_value=1, step=1, value=1)

        # Liste für die hochgeladenen Dateien
        uploaded_files = []

        # Datei-Uploads basierend auf der Anzahl der angegebenen Dateien
        for i in range(num_files):
            uploaded_file = st.file_uploader(f"Please upload file {i+1} (PDF or Excel)", type=['xlsx', 'pdf'], key=f'file_uploader_{i}')
            if uploaded_file:
                uploaded_files.append(uploaded_file)

        # Überprüfen, ob alle Dateien hochgeladen wurden
        if len(uploaded_files) == num_files:
            all_dfs = []  # Liste zum Speichern der DataFrames

            for file in uploaded_files:
                if file.name.endswith('.xlsx'):
                    extracted_df = process_excel(file, synonyms)
                elif file.name.endswith('.pdf'):
                    text = process_pdf(file)
                    extracted_df = text_extraction(text, synonyms)
                
                all_dfs.append(extracted_df.T)
            
            #st.write(all_dfs)

            # Alle DataFrames zusammenführen
            if all_dfs:

                max_col_number = sum([df.shape[1] for df in all_dfs])
                combined_df = pd.concat(all_dfs, axis=1)
                combined_df.columns = range(max_col_number)
                st.session_state['values'] = combined_df
                st.write("Combined DataFrame:")

                # Verwende die lesbaren Feature-Namen für die Anzeige
                display_df = st.session_state['values'].rename(index=feature_names)
                st.write(display_df)




        if 'values' in st.session_state:
            Input_modify = st.radio("Do you want to change the imported data?", ("no", "yes"))
            if Input_modify == "yes": 
            
                # Erstelle ein DataFrame mit den importierten Werten
                editable_df = st.session_state['values'].copy()

                features_names = ['AgeOnInclusion', 'RRSysWeightedMeanValue', 'RRDiaWeightedMeanValue',
                                    'sO2WeightedMeanValue', 'PHWeightedMean', 'LactateWeightedMean',
                                    'gluWeightedMean', 'HCO3WeightedMean', 'DayNumber',
                                    'PO2WeightedMean', 'PCO2WeightedMean', 'HBWeightedMean',
                                    'leucoWeightedMean', 'ureaWeightedMean', 'HRWeightedMean',
                                    'TempWeightedMean', 'NaWeightedMean', 'KWeightedMean', 'ClWeightedMean',
                                    'Height', 'Weight', 'plateletsWeightedMean']
                
                # Prüfe, welche Spalten in editable_df fehlen
                missing_cols = [col for col in features_names if col not in editable_df.index]

                # Füge fehlende Indizes mit NaN-Werten hinzu
                if missing_cols:
                    for col in missing_cols:
                        editable_df.loc[col] = None

                
                # Index in eine reguläre Spalte umwandeln
                editable_df = editable_df.reset_index()
                editable_df.columns = ['Feature'] + editable_df.columns[1:].tolist()

            

                # Stelle eine interaktive Tabelle bereit, in der der Benutzer die Werte ändern kann
                edited_df = st.data_editor(editable_df, num_rows="dynamic")

                edited_df = edited_df.set_index('Feature')
                
                # Speichere die bearbeiteten Daten zurück in den Session State
                st.session_state['values'] = edited_df
                        

                        
                weighted_means_df = calculate_weighted_means(st.session_state['values']).T
                weighted_means_df = weighted_means_df[features_names]
                weighted_means_df = weighted_means_df.astype(float)

                        # Ergebnis für das Modell vorbereiten
                st.write("Berechnete Werte, die ans Modell geschickt werden:")
                st.write(weighted_means_df.T)

                        # Speichern in Session State
                st.session_state['weighted_means_df'] = weighted_means_df
                st.session_state['data_formatted'] = True

            
            if Input_modify == "no": 
                
                # Erstelle ein DataFrame mit den importierten Werten
                editable_df = st.session_state['values'].copy()

                features_names = ['AgeOnInclusion', 'RRSysWeightedMeanValue', 'RRDiaWeightedMeanValue',
                                    'sO2WeightedMeanValue', 'PHWeightedMean', 'LactateWeightedMean',
                                    'gluWeightedMean', 'HCO3WeightedMean', 'DayNumber',
                                    'PO2WeightedMean', 'PCO2WeightedMean', 'HBWeightedMean',
                                    'leucoWeightedMean', 'ureaWeightedMean', 'HRWeightedMean',
                                    'TempWeightedMean', 'NaWeightedMean', 'KWeightedMean', 'ClWeightedMean',
                                    'Height', 'Weight', 'plateletsWeightedMean']
                
                # Prüfe, welche Spalten in editable_df fehlen
                missing_cols = [col for col in features_names if col not in editable_df.index]

                # Füge fehlende Indizes mit NaN-Werten hinzu
                if missing_cols:
                    for col in missing_cols:
                        editable_df.loc[col] = None

                # Stelle eine interaktive Tabelle bereit, in der der Benutzer die Werte ändern kann
                edited_df = editable_df.copy()
                
                # Speichere die bearbeiteten Daten zurück in den Session State
                st.session_state['values'] = edited_df
                        

                        
                weighted_means_df = calculate_weighted_means(st.session_state['values']).T
                weighted_means_df = weighted_means_df[features_names]
                weighted_means_df = weighted_means_df.astype(float)

                        # Ergebnis für das Modell vorbereiten
                st.write("Berechnete Werte, die ans Modell geschickt werden:")
                st.write(weighted_means_df.T)

                        # Speichern in Session State
                st.session_state['weighted_means_df'] = weighted_means_df
                st.session_state['data_formatted'] = True



    if User_Input == "Manual input":        

        # Mapping von technischen Namen zu verständlichen Bezeichnungen
        feature_names = {
            'RRSysWeightedMeanValue': 'Systolic Blood Pressure',
            'RRDiaWeightedMeanValue': 'Diastolic Blood Pressure',
            'sO2WeightedMeanValue': 'Oxygen Saturation',
            'TempWeightedMean': 'Body Temperature',
            'PHWeightedMean': 'Blood pH',
            'LactateWeightedMean': 'Blood Lactate Level',
            'gluWeightedMean': 'Blood Glucose Level',
            'HCO3WeightedMean': 'Bicarbonate (HCO3)',
            'PO2WeightedMean': 'Partial Pressure of Oxygen (PaO2)',
            'PCO2WeightedMean': 'Partial Pressure of Carbon Dioxide (PaCO2)',
            'HBWeightedMean': 'Hemoglobin (HB)',
            'leucoWeightedMean': 'Leukocytes',
            'ureaWeightedMean': 'Blood Urea',
            'HRWeightedMean': 'Heart Rate',
            'NaWeightedMean': 'Sodium (Na+)',
            'KWeightedMean': 'Potassium (K+)',
            'ClWeightedMean': 'Chloride (Cl-)',
            'plateletsWeightedMean': 'Platelets'
        }

        # Umgekehrtes Mapping, um später wieder auf die technischen Namen zuzugreifen
        reverse_feature_names = {v: k for k, v in feature_names.items()}

        # Feature-Liste und Anzahl der Stunden (24 Stunden)
        features = ['AgeOnInclusion', 'Height', 'Weight', 'DayNumber'] + list(feature_names.keys())

        hours = [f"{i}h" for i in range(1, 25)]

        # Erstelle ein DataFrame für die Eingabe, initialisiert mit NaN
        input_data = pd.DataFrame(np.nan, index=[feature_names[f] for f in feature_names], columns=hours)

        # Erster Teil der Features: Einfache Eingabefelder
        st.write("Please enter the values for the following features:")

        age = st.number_input("AgeOnInclusion", min_value=0, max_value=120, value=30)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        day_number = st.number_input("Day Number on the Intensive Care Unit", min_value=0, max_value=30, value=1)

        # Zweiter Teil: Tabelle für stündliche Werteingabe
        st.write("Please enter the hourly values for the following features:")

        # Interaktive Eingabetabelle anzeigen
        input_table = st.data_editor(input_data, num_rows="dynamic")

        # Funktion zur Berechnung des gewichteten Mittels
        def calculate_weighted_mean(df):
            weighted_means = {}
            for feature in df.index:
                values = df.loc[feature].values
                last_valid_value = None
                weighted_sum = 0
                total_weight = 0
                for i, value in enumerate(values):
                    if not np.isnan(value):
                        if last_valid_value is not None:
                            weighted_sum += last_valid_value * (i + 1 - total_weight)
                            total_weight += (i + 1 - total_weight)
                        last_valid_value = value
                # Add remaining period until end if last_valid_value exists
                if last_valid_value is not None:
                    weighted_sum += last_valid_value * (len(values) - total_weight)
                    total_weight += (len(values) - total_weight)
                weighted_mean = weighted_sum / total_weight if total_weight > 0 else np.nan
                weighted_means[reverse_feature_names[feature]] = weighted_mean
            return weighted_means

        # Berechnung der gewichteten Mittelwerte für jedes Feature
        weighted_means_df = None
        
        st.session_state['Berechne_Weighted_Means'] = False

        #if st.button("Berechne Weighted Means"):
        #    st.session_state['Berechne_Weighted_Means'] = True

        #if st.session_state['Berechne_Weighted_Means'] == True:
        weighted_means = calculate_weighted_mean(input_table)


        # DataFrame mit den berechneten weighted means erstellen
        weighted_means_df = pd.DataFrame(weighted_means, index=["Patient"])

        # Füge die einzelnen Eingabefelder hinzu
        weighted_means_df['AgeOnInclusion'] = age
        weighted_means_df['Height'] = height
        weighted_means_df['Weight'] = weight
        weighted_means_df['DayNumber'] = day_number
        
        col_names = ['AgeOnInclusion', 'RRSysWeightedMeanValue', 'RRDiaWeightedMeanValue',
                    'sO2WeightedMeanValue', 'PHWeightedMean', 'LactateWeightedMean',
                    'gluWeightedMean', 'HCO3WeightedMean', 'DayNumber',
                    'PO2WeightedMean', 'PCO2WeightedMean', 'HBWeightedMean',
                    'leucoWeightedMean', 'ureaWeightedMean', 'HRWeightedMean',
                    'TempWeightedMean', 'NaWeightedMean', 'KWeightedMean', 'ClWeightedMean',
                    'Height', 'Weight', 'plateletsWeightedMean']
        
        weighted_means_df = weighted_means_df[col_names]

        # Ergebnis für das Modell vorbereiten
        st.write("Calculated values to be sent to the model:")
        st.write(weighted_means_df.T)

        # Speichern in Session State
        st.session_state['weighted_means_df'] = weighted_means_df
        st.session_state['data_formatted'] = True


        # Du kannst hier die weiteren Schritte wie Vorhersagen und Counterfactuals implementieren...

    if  st.session_state['data_formatted'] == True:
        if  st.button("Predict mortality"):


            model = st.session_state['model']
            #if 'data_formatted' in st.session_state:
            # Vorhersage durchführen
            if 'weighted_means_df' in st.session_state:
                st.header("**Results**")
                
                weighted_means_df = st.session_state['weighted_means_df']
                #print(weighted_means_df)
                prediction,prediction_proba = model.predict(weighted_means_df)
                #prediction_proba = model.predict_proba(weighted_means_df)
                #print(prediction, prediction_proba)
                
                st.header("**How to interprete the results:**")
                
                st.write("**Prediction Value = 1** indicates that the patient is predicted to be **deceased**.")
                st.write("**Prediction Value = 0** indicates that the patient is predicted to be **alive**.")
                #st.write("**Cohort 1** corresponds to predictions with **high accuracy**.")
                #st.write("**Cohort 2** corresponds to predictions with **lower accuracy**.")
            else:
                st.error("Please calculate the Weighted Means first!")
                    
        if  st.button("SHAP explanations"):
                    
            # Add a button to generate the SHAP plot
            if 'weighted_means_df' in st.session_state:
                weighted_means_df = st.session_state['weighted_means_df']
                model.plot_shap_waterfall(weighted_means_df)
            else:
                st.error("Please calculate the Weighted Means first!")

        if st.button("Generate Counterfactuals"):
            st.session_state['counterfactuals_button'] = True

        if 'counterfactuals_button' in st.session_state:
            if 'weighted_means_df' in st.session_state:
                weighted_means_df = st.session_state['weighted_means_df']

                col = [
                    'RRSysWeightedMeanValue',
                    'RRDiaWeightedMeanValue',
                    'sO2WeightedMeanValue',
                    'PHWeightedMean',
                    'LactateWeightedMean',
                    'gluWeightedMean',
                    'HCO3WeightedMean',
                    'PO2WeightedMean',
                    'PCO2WeightedMean',
                    'HBWeightedMean',
                    'leucoWeightedMean',
                    'ureaWeightedMean',
                    'HRWeightedMean',
                    'TempWeightedMean',
                    'NaWeightedMean',
                    'KWeightedMean',
                    'ClWeightedMean',
                    'plateletsWeightedMean']

                permitted_range = {
                    'RRSysWeightedMeanValue': [90, 150],
                    'RRDiaWeightedMeanValue': [50, 90],
                    'sO2WeightedMeanValue': [90, 100],
                    'PHWeightedMean': [7.35, 7.45],
                    'LactateWeightedMean': [0, 2],
                    'gluWeightedMean': [60, 200],
                    'HCO3WeightedMean': [22, 26],
                    'PO2WeightedMean': [40, 100],
                    'PCO2WeightedMean': [24, 40],
                    'HBWeightedMean': [8, 15],
                    'ureaWeightedMean': [15, 50],
                    'HRWeightedMean': [60, 100],
                    'TempWeightedMean': [36.5, 38],
                    'NaWeightedMean': [130, 150],
                    'KWeightedMean': [3.5, 5],
                    'ClWeightedMean': [90, 110],
                    'leucoWeightedMean': [5, 20],
                    'plateletsWeightedMean': [150, 450]}
                

                # Benutzer wählt die Features aus, die er ändern möchte
                st.write("Select the features you want to modify:")
                selected_columns = st.multiselect("Features", options=col)

                if selected_columns:
                    user_permitted_range = {}
                    # Benutzer gibt den Bereich für jedes ausgewählte Feature ein
                    for feature in selected_columns:
                        min_val, max_val = st.slider(
                            f"Choose the range for {feature}",
                            min_value=float(permitted_range[feature][0]),
                            max_value=float(permitted_range[feature][1]),
                            value=(float(permitted_range[feature][0]), float(permitted_range[feature][1]))
                        )
                        user_permitted_range[feature] = [min_val, max_val]
                        
                    total_CFs = st.number_input("How many counterfactuals would you like to generate?", min_value=1, value=5, step=1)
                
                    # Counterfactuals generieren, wenn Features ausgewählt wurden
                    
                    if st.button("show Counterfactuals"):

                        if selected_columns:
                            counterfactuals_df = model.generate_counterfactuals(
                                weighted_means_df,
                                total_CFs=total_CFs,
                                desired_class=0,
                                features_to_vary=selected_columns,
                                permitted_range=user_permitted_range
                            )
                            st.write("Counterfactuals:")
                            st.write(counterfactuals_df)
                            
                            # Zeige nur die geänderten Werte
                            #changes_df = counterfactuals_df[selected_columns].copy()
                            aligned_counterfactuals_df, aligned_weighted_means_df = counterfactuals_df.align(weighted_means_df.iloc[0], axis=1, copy=False)
                            differences = aligned_counterfactuals_df != aligned_weighted_means_df
                            changed_values_df = counterfactuals_df[differences]

                            # Zeige die geänderten Werte
                            st.write("Values to change:")
                            st.write(changed_values_df)
                            
                            
                        else:
                            st.error("Choose at least one feature to change!")
            else:
                st.error("Please, calculate firstly weighted means!")
                    
                



