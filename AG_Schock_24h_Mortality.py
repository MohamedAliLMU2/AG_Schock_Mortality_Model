
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
            with zip_ref.open('pca_model_1.pkl') as file:
                t_model = pickle.load(file)
            
            with zip_ref.open('pca_model_2.pkl') as file:
                cohort1_model = pickle.load(file)
    
            with zip_ref.open('pca_scaler.pkl') as file:
                scaler = pickle.load(file)
    
            with zip_ref.open('pca_model_cohrot_2.pkl') as file:  # corrected typo
                cohort2_model = pickle.load(file)
    
            with zip_ref.open('pca_model_exp1.pkl') as file:
                exp1 = pickle.load(file)
    
            with zip_ref.open('pca_model_exp2.pkl') as file:
                exp2 = pickle.load(file)
                
                
            with zip_ref.open('pca_model_1_3d.pkl') as file:
                t_model_3d = pickle.load(file)
            
            with zip_ref.open('pca_model_2_3d.pkl') as file:
                cohort1_model_3d = pickle.load(file)
    
            with zip_ref.open('pca_scaler_3d.pkl') as file:
                scaler_3d = pickle.load(file)
    
            with zip_ref.open('pca_model_cohrot_2_3d.pkl') as file:  # corrected typo
                cohort2_model_3d = pickle.load(file)
    
            with zip_ref.open('pca_model_exp1_3d.pkl') as file:
                exp1_3d = pickle.load(file)
    
            with zip_ref.open('pca_model_exp2_3d.pkl') as file:
                exp2_3d = pickle.load(file)
    
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
        def __init__(self, t_model=t_model, cohort1_model=cohort1_model, scaler=scaler, cohort2_model=cohort2_model, exp_dice1=exp1, exp_dice2=exp2,
                     t_model_3d=t_model_3d, cohort1_model_3d=cohort1_model_3d, scaler_3d=scaler_3d, cohort2_model_3d=cohort2_model_3d, exp_dice1_3d=exp1_3d, exp_dice2_3d=exp2_3d):
            self.t_model = t_model
            self.cohort1_model = cohort1_model
            self.scaler = scaler
            self.cohort2_model = cohort2_model
            self.exp1 = exp_dice1
            self.exp2 = exp_dice2
            
            self.t_model_3d = t_model_3d
            self.cohort1_model_3d = cohort1_model_3d
            self.scaler_3d = scaler_3d
            self.cohort2_model_3d = cohort2_model_3d
            self.exp1_3d = exp_dice1_3d
            self.exp2_3d = exp_dice2_3d

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
                    y_pred = self.cohort1_model.predict(X_test_full)

                    st.write("The 24h-Mortality ist predicted with 0.91 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next day:", y_pred[0])
                    
                    y_pred_proba = self.cohort1_model.predict_proba(X_test_full)[:, 1]
                    st.write("Mortality probability  for the next day:", y_pred_proba[0])


                if pred == 1:
                    y_pred = self.cohort2_model.predict(X_test_full)

                    st.write("The 24h-Mortality ist predicted with 0.72 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next day:", y_pred[0])
                    
                    y_pred_proba = self.cohort2_model_3d.predict_proba(X_test_full)[:, 1]
                    st.write("Mortality probability  for the next day:", y_pred_proba[0])
                    
                    
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
                    y_pred = self.cohort1_model_3d.predict(X_test_full)

                    st.write("The 3d-Mortality ist predicted with 0.81 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next 3 days:", y_pred[0])

                    y_pred_proba = self.cohort1_model_3d.predict_proba(X_test_full)[:, 1]
                    st.write("Mortality probability  for the next 3 days:", y_pred_proba[0])

                if pred == 1:
                    y_pred = self.cohort2_model_3d.predict(X_test_full)

                    st.write("The 3d-Mortality ist predicted with 0.34 ROC AUC (Accuracy)")
                    st.write("Mortality prediction for the next 3 days:", y_pred[0])

                    y_pred_proba = self.cohort2_model_3d.predict_proba(X_test_full)[:, 1]
                    st.write("Mortality probability  for the next 3 days:", y_pred_proba[0])


            return y_pred, y_pred_proba

        
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



    # Modell initialisieren
    model = model_24h_mortality(t_model, cohort1_model, scaler, cohort2_model, exp1, exp2)
    st.session_state['model'] = model
    
    st.session_state['data_formatted'] = False

    User_Input = st.radio("How do you prefer to input the data?", ("Automatic input form pdf/excel", "Manual input"))
    if User_Input == "Automatic input form pdf/excel":        
        
        # Synonym-Mapping für die Features in Deutsch und Englisch
        synonyms = {
            'AgeOnInclusion': ['Age', 'Alter', 'Years', 'Years of Age', 'Lebensalter'],
            'RRSysWeightedMeanValue': ['systolischer Blutdruck', 'systolic blood pressure', 'systolic BP', 'blood pressure systolic', 'systolische Blutdruckmessung', 'RR systolisch', 'RRsys'],
            'RRDiaWeightedMeanValue': ['diastolischer Blutdruck', 'diastolic blood pressure', 'diastolic BP', 'blood pressure diastolic', 'diastolische Blutdruckmessung', 'RR diastolisch', 'RRdia'],
            'sO2WeightedMeanValue': ['sauerstoffsättigung', 'oxygen saturation', 'SpO2', 'sO2', 'oxygen level', 'O2 sat', 'Sauerstofflevel', 'O2-Sättigung'],
            'TempWeightedMean': ['körpertemperatur', 'body temperature', 'temp', 'temperature', 'core temperature', 'Körperwärme', 'Körpertemp'],
            'PHWeightedMean': ['ph', 'blood pH', 'pH level', 'pH-Wert', 'Säure-Basen-Status', 'Blut-pH'],
            'LactateWeightedMean': ['laktatspiegel', 'lactate level', 'blood lactate', 'Laktat', 'Lactate', 'Serumlaktat', 'Laktatkonzentration'],
            'gluWeightedMean': ['blutzucker', 'blood glucose', 'glucose', 'Blutglukose', 'Blutzuckerspiegel', 'blood sugar', 'Glucoselevel', 'Glukosespiegel'],
            'HCO3WeightedMean': ['bicarbonat', 'bicarbonate', 'HCO3', 'Bikarbonatspiegel', 'Hydrogencarbonate', 'Serum-Bicarbonat', 'HCO₃⁻'],
            'DayNumber': ['Day number', 'Tag auf ITS', 'Hospital Day', 'Stay Day', 'Hospital Day Number', 'Verweildauer', 'Tage seit Aufnahme', 'ITS-Tag'],
            'PO2WeightedMean': ['sauerstoffpartialdruck', 'partial pressure of oxygen', 'PaO2', 'pO2', 'Oxygen partial pressure', 'Oxygen Tension', 'arterieller Sauerstoffdruck', 'Sauerstoffdruck'],
            'PCO2WeightedMean': ['kohlendioxidpartialdruck', 'partial pressure of carbon dioxide', 'PaCO2', 'pCO2', 'Carbon dioxide partial pressure', 'arterieller Kohlendioxid-Druck', 'CO2-Druck', 'CO₂-Partialdruck'],
            'HBWeightedMean': ['hämoglobin', 'hemoglobin', 'Hb', 'blood hemoglobin', 'Hämoglobinspiegel', 'Hb-Wert', 'Bluthämoglobin'],
            'leucoWeightedMean': ['leukozyten', 'leukocytes', 'white blood cells', 'WBC', 'Leukocytenzahl', 'Weiße Blutkörperchen', 'Leukos'],
            'ureaWeightedMean': ['harnstoff', 'urea', 'blood urea', 'Urea nitrogen', 'Harnstoffstickstoff', 'BUN', 'Harnstoffkonzentration'],
            'HRWeightedMean': ['herzfrequenz', 'heart rate', 'pulse', 'HR', 'Pulsfrequenz', 'Puls', 'Cardiac rate', 'Herzschlagfrequenz'],
            'NaWeightedMean': ['natrium', 'sodium', 'Na', 'Serumnatrium', 'Sodium level', 'Natriumspiegel', 'Natriumwert'],
            'KWeightedMean': ['kalium', 'potassium', 'K', 'Serumkalium', 'Potassium level', 'Kaliumspiegel', 'Kaliumwert'],
            'ClWeightedMean': ['chlorid', 'chloride', 'Cl', 'Serumchlorid', 'Chloridspiegel', 'Chloridion', 'Chloridwert'],
            'Height': ['Height', 'Größe', 'Body height', 'Körpergröße', 'Stature', 'Länge', 'Körperlänge'],
            'Weight': ['Weight', 'Gewicht', 'Body weight', 'Körpergewicht', 'Mass', 'Masse'],
            'plateletsWeightedMean': ['thrombozyten', 'platelets', 'Thrombocytes', 'Plättchen', 'Platelet count', 'Thrombozytenzahl', 'Blutplättchen']
        }


        # Funktion zur Verarbeitung der hochgeladenen Excel-Datei
        def process_excel(file):
            df = pd.read_excel(file)

            df = df.dropna(how='all')  # Entfernt leere Zeilen
            df = df.dropna(axis=1, how='all')  # Entfernt leere Spalten
            

            text = df.to_string(index=False, na_rep='')  
            text = text.replace('Unnamed: 0', '')

            text = '\n'.join(line.strip() for line in text.splitlines())

            
            return text




        # Funktion zur Verarbeitung der hochgeladenen PDF-Datei
        def process_pdf(file):
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            return text



        def text_extraction(text, synonyms):
            # Muster für Zeitangaben im Format HH:MM:SS oder HH:MM
            time_pattern = r'\b\d{2}:\d{2}(?::\d{2})?\b'
            
            # Finde alle Zeitangaben im Text
            times = re.findall(time_pattern, text)
            
            # Konvertiere Zeiten zu HH:MM:SS Format (falls notwendig)
            def format_time(time_str):
                if len(time_str) == 5:  # Wenn Zeit im Format HH:MM vorliegt
                    return time_str + ":00"
                return time_str  # Bereits im Format HH:MM:SS
        
            times = [format_time(t) for t in times]
        
            # Datenstruktur für die Ergebnisse
            data = {'Time': times}
        
            # Durch die Synonyme iterieren und Features extrahieren
            for feature, synonym_list in synonyms.items():
                values = []
                for synonym in synonym_list:
                    # Muster für das jeweilige Feature: Wert nach dem Synonym (korrigiert für ganze und dezimale Zahlen)
                    pattern = rf'{re.escape(synonym)}\s*(\d+\.?\d*)'
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        # Die Werte in die entsprechende Spalte einfügen
                        values.extend([float(match) for match in matches])
                        break  # Feature wurde gefunden, keine weiteren Synonyme prüfen
        
                # Falls Werte vorhanden sind, in die Datenstruktur einfügen
                if values:
                    # Erstelle eine Liste von Werten, die den gefundenen Zeiten zugeordnet sind
                    data[feature] = [values[i] if i < len(values) else None for i in range(len(times))]
        
            # In eine Tabelle umwandeln (DataFrame)
            df = pd.DataFrame(data)
        
            # Verstecke leere Spalten, die keine Daten enthalten
            df = df.dropna(axis=1, how='all')
        
            return df





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


        # Funktion zur Normalisierung von Zeit
        def normalize_time(text):

            time_pattern = r'(\d{1,2})[:.](\d{2})'
            matches = re.findall(time_pattern, text)
            normalized_times = []
            for match in matches:
                hours, minutes = match
                normalized_time = f"{int(hours):02d}:{int(minutes):02d}"
                normalized_times.append(normalized_time)
            return normalized_times
        
        # Funktion zur Erkennung von Synonymen und Ersetzen durch Standardbegriffe
        def replace_synonyms(text, synonyms):
            for standard_term, synonym_list in synonyms.items():
                for synonym in synonym_list:
                    text = re.sub(rf'\b{re.escape(synonym)}\b', standard_term, text, flags=re.IGNORECASE)
            return text
        
        # Funktion zum Verarbeiten der Daten und Erstellen eines DataFrames
        def text_extraction(input_text, synonyms):

            # Schritt 1: Normalisiere die Uhrzeiten
            times = normalize_time(input_text)
        
            # Schritt 2: Ersetze die Synonyme im Text
            normalized_text = replace_synonyms(input_text, synonyms)
        
            # Schritt 3: Extrahiere die Werte und Zuordnungen
            # Diese Annahme basiert darauf, dass die Daten im Format "Begriff Wert" vorliegen
            data_pattern = r'(\w+(?: \w+)?)(?:\s+(\d+))'  # Muster zum Erkennen von "Begriff Wert"
            data = re.findall(data_pattern, normalized_text)
        
            # Schritt 4: Erstelle den DataFrame
            columns = ['Zeit'] + [term for term, _ in data[:len(times)]]
            values = [times] + [[int(value) for _, value in data[:len(times)]]]
        
            df = pd.DataFrame([values], columns=columns)
        
            return df




        #if st.button("Data input as PDF / Excel"):

            # Datei-Upload-Optionen
        uploaded_file = st.file_uploader("Please upload a PDF or Excel file", type=['xlsx', 'pdf'])

            # Initialisiere die Session-State-Variable, um Daten zu speichern
        #if 'weighted_means' not in st.session_state:
        #    st.session_state['weighted_means'] = {}



        if uploaded_file:
            if uploaded_file.name.endswith('.xlsx'):
                text1 = process_excel(uploaded_file)
            elif uploaded_file.name.endswith('.pdf'):
                text1 = process_pdf(uploaded_file)
            st.write(text1)
            extracted_df = text_extraction(text1, synonyms)
            st.write(extracted_df.T)

            # Aktualisiere Session State
            #if 'values' in st.session_state:
            #    st.session_state['values'] = pd.concat([st.session_state['values'], extracted_df.T])
            #else:
            st.session_state['values'] = extracted_df.T
            

            # Frage nach weiteren Dateien
            more_files = st.radio("Do you want to upload more data?", ("Yes", "No"))
            if more_files == "Yes":
                uploaded_file2 = st.file_uploader("Please upload a PDF or an Excel file", type=['xlsx', 'pdf'])

                if uploaded_file2:
                    if uploaded_file2.name.endswith('.xlsx'):
                        df = process_excel(uploaded_file2)
                        text = df
                    elif uploaded_file2.name.endswith('.pdf'):
                        text = process_pdf(uploaded_file2)
                    #st.write(text)
                    extracted_df = text_extraction(text, synonyms)
                    max_col_number = max([int(col) for col in st.session_state['values'].columns])
                    new_columns = {col: str(max_col_number + i + 1) for i, col in enumerate(extracted_df.T.columns)}
                    extracted_df = extracted_df.T.rename(columns=new_columns)

                    st.session_state['values'] = pd.concat([st.session_state['values'], extracted_df], axis=1)
                    st.write(st.session_state['values'])

            #if more_files == "Nein":
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


            if more_files == "No":
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
                    
                


    #excel import reparieren #seitenübergang reparieren # 3d model # zeit zum sterben model #LLM
