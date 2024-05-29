import streamlit as st 
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Fungsi untuk preprocessing dataset
def dataset_preprocessing(main_df):
    kolom_dihapus_main_df = ['EmployeeId', 'DailyRate', 'Gender', 'HourlyRate', 'MaritalStatus', 'OverTime', 'PercentSalaryHike', 'StandardHours', 'StockOptionLevel', 'TrainingTimesLastYear', 'YearsSinceLastPromotion']
    dataset_ml = main_df.drop(kolom_dihapus_main_df, axis=1)

    fitur_normalisasi, fitur_encoding = [], []

    for fitur in dataset_ml:
        if dataset_ml[fitur].dtype == "object":
            fitur_encoding.append(fitur)
        else:
            fitur_normalisasi.append(fitur)

    # Melakukan Label Encoder pada fitur
    LE = LabelEncoder() #Mendefenisikan LabelEncoder sebagai LE
    dataset_ml_main = dataset_ml.copy() #Mencegah SettingWithCopyWarning pandas
    dataset_ml_main[fitur_encoding] = dataset_ml_main[fitur_encoding].apply(LE.fit_transform) #Menerapkan LabelEncoder untuk fitur terpilih

    # Melakukan normalisasi untuk beberapa fitur
    scaler = MinMaxScaler()  #Mendefinisikan MinMaxScaler
    dataset_ml_main[fitur_normalisasi] = scaler.fit_transform(dataset_ml_main[fitur_normalisasi]) #Menerapkan fit_transform untuk normalisasi fitur terpilih

    dataset_ml_main.drop("Attrition", axis=1, inplace=True)

    return dataset_ml_main


# Fungsi untuk prediksi attrition
def predict_attrition(fix_main_df):

    best_model = joblib.load("random_forest.joblib")

    prediction = best_model.predict(fix_main_df)

    return prediction


# Fungsi untuk menampilkan hasil prediksi
def result_attrition(main_df, prediction_array):

    result_df_1 = main_df["EmployeeId"]
    result_df_2 = pd.DataFrame(data=prediction_array)
    result_df = pd.merge(result_df_1,result_df_2, how='left', left_index = True, right_index = True)
    result_df.columns = ["ID Karyawan", "Prediksi Status Karyawan"]

    result_df_fix = result_df.copy()
    result_df_fix["Prediksi Status Karyawan"] = result_df_fix["Prediksi Status Karyawan"].apply(lambda x: "Bertahan" if x == 0 else "Keluar")

    return result_df_fix



# Fungsi utama website  
def main():

    st.title('Jaya Jaya Maju')

    st.header('Dasbor Karyawan')

    st.text('Halo, Selamat datang di halaman dasbor prediksi karyawan keluar/bertahan dari Perusahaan Jaya Jaya Maju')

    with st.expander("Kenalan Dulu dengan Jaya Jaya Maju..."):
        st.write(
            """
                Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. 
                Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri.
            """
        )

    with st.expander("Mengapa dasbor ini dibuat ya?"):
        st.write(
            """
                Sebagai perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. 
                Hal ini berimbas tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%.
            """
        )

    with st.expander("Bagaimana sih cara menggunakan dasbor ini?"):
        st.write(
            """
                Sederhana kok, kamu bisa ngikutin tahapan-tahapan dibawah ini:
                1. Masukan dulu dataset yang sesuai pada menu "Masukan Dataset" dibawah. Dataset-nya dimana? Bisa didownload dari github/hasil output notebook.ipynb dengan nama file "df_employee_for_ML.csv"
                2. Kalo datasetnya sudah muncul, mantapss...
                3. Tahan duiu, kamu bisa set berapa banyak data yang mau di prediksi melalui menu slider ya...
                4. Yuk lanjut!!! kalau udah cocok berapa banyak datanya, gaskun klik tombol "Prediksi Status Karyawan"
                5. Eits, yang sabar ya tunggu proses prediksi-nya...
                6. Kalau sudah muncul hasil prediksinya, yeeey Selamat ya Kamu Berhasil...
            """
        )

    st.write("##")

    uploaded_files = st.file_uploader("Masukan Dataset", accept_multiple_files=True, key="fileuploader1")
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("filename:", uploaded_file.name)
        try:
            row_value_df = st.slider('Tampilkan Data', 0, df.shape[0], 5) 
            df = df.iloc[:row_value_df]
            st.dataframe(df)
        except NameError:
            st.write("")

    # when 'Predict' is clicked, make the prediction and store it
    try:
        if uploaded_files is not None:
            try:
                if st.button("Prediksi Status Karyawan"): 
                    main_new_dataset = dataset_preprocessing(df) 
                    
                    result = predict_attrition(main_new_dataset)

                    result_dataset = result_attrition(df, result)

                    st.success(f"Hasil Prediksi Berhasil untuk {result_dataset.shape[0]} karyawan perusahaan Jaya Jaya Maju")
                    row_value_result_dataset = st.slider('Tampilkan Data Hasil Prediksi', 0, result_dataset.shape[0], result_dataset.shape[0])
                    result_dataset = result_dataset.iloc[:row_value_result_dataset]
                    st.dataframe(result_dataset)
                    
            except ValueError:
                    st.error('Dataset Kosong! Pastikan Dataset memiliki isi sebelum di prediksi', icon="🔥")
                    
    except UnboundLocalError:
        st.error('Anda belum menginput dataset', icon="🚨")

        


if __name__=='__main__': 
    main()


st.caption('Copyright (c) 2024')

