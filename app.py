from streamlit_option_menu import option_menu
import joblib
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

@st.cache_resource
def get_driver():
    return webdriver.Chrome(
        service=Service(
            ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        ),
        options=options,
    )
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://lh5.googleusercontent.com/p/AF1QipNgmGyncJl5jkHg6gnxdTSTqOKtIBpy-kl9PgDz=w540-h312-n-k-no',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">Analisi Sentimen Wisata Madura Dengan Maximum Entropy</h2></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"></h3>""",unsafe_allow_html=True), 
        ["Home", "Dataset","prediksi ulasan","Implementation"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
    if selected == "Home" :
        st.write("""<h3 style="text-align: center;">
        <img src="https://lh5.googleusercontent.com/p/AF1QipNgmGyncJl5jkHg6gnxdTSTqOKtIBpy-kl9PgDz=w540-h312-n-k-no" width="500" height="300">
        </h3>""", unsafe_allow_html=True)
    if selected == "Dataset":
        st.write("Data Sebelum Preprocessing")
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        st.write(data['Ulasan'].head(10))
        st.write("Data Setelah Preprocessing")
        file_path2 = 'data preprocessing.csv'  # Ganti dengan path ke file Anda
        data2 = pd.read_csv(file_path2)
        st.write(data2['Ulasan'].head(10))
    if selected == "prediksi ulasan":
        import joblib
        # Menggunakan pandas untuk membaca file CSV
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(data['stopword']).toarray()
        loaded_model = joblib.load('final_maxent_model.pkl')
        loaded_vectorizer = joblib.load('tfidf (1).pkl')
        with st.form("my_form"):
            st.subheader("Implementasi")
            ulasan = st.text_input('Masukkan ulasan')  # Input ulasan dari pengguna
            submit = st.form_submit_button("Prediksi")
            if submit:
                if ulasan.strip():  # Validasi input tidak kosong
                    # Transformasikan ulasan ke bentuk vektor
                    new_X = vectorizer.transform([ulasan]).toarray()
        
                    # Membuat dictionary dengan nama feature sesuai format model
                    new_data_features = {f"feature_{j}": new_X[0][j] for j in range(new_X.shape[1])}
                    
                    # Prediksi menggunakan model
                    new_pred = loaded_model.classify(new_data_features)
        
                    # Tampilkan hasil prediksi
                    st.subheader('Hasil Prediksi')
                    st.write(f"Prediction for New Data: {new_pred}")
                else:
                    st.error("Masukkan ulasan terlebih dahulu!")

    if selected == "Implementation":
        import joblib
        # Menggunakan pandas untuk membaca file CSV
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(data['stopword']).toarray()
        loaded_model = joblib.load('final_maxent_model.pkl')
        loaded_vectorizer = joblib.load('tfidf (1).pkl')


    
        st.subheader("Implementasi Menggunakan Data Baru")
        url = "https://colab.research.google.com/drive/1im2fPYWSGElnmKdOR8ysApZsiG_jOlRR?usp=sharing"
        st.write("Code implementasi scrapping dengan selenium [link](%s)" % url)
        url2 = "https://github.com/feb11fff/sistem-skripsi/tree/main"
        st.write("source code implemnatasi streamlit pada github [link](%s)" % url2)
        st.title("pilih sentimen wisata")
        
      
        if st.button("Bukit Jaddih"):
            try:
                from bs4 import BeautifulSoup
                from selenium import webdriver
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.common.exceptions import NoSuchElementException, WebDriverException
                from selenium.webdriver.chrome.service import Service
                import os
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--no-sandbox")
                
                driver = webdriver.Chrome(options=options)
                # URL dari Google Search
                url = 'https://www.google.com/maps/place/Jaddih+Hill+Madura/@-7.0822777,112.7569647,17z/data=!4m8!3m7!1s0x2dd8045eb0acb79d:0x4a24af02fd796f55!8m2!3d-7.082283!4d112.7595396!9m1!1b1!16s%2Fg%2F11c2r8kctr?entry=ttu&hl=id&gl=ID'
                driver.get(url)
                time.sleep(5)
                tombol = driver.find_element(By.XPATH, "//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[7]/div[2]/button")
                tombol.click()
                time.sleep(5)
                # Klik menggunakan JavaScript
                terbaru_button = driver.find_element(By.XPATH, "//div[@class='mLuXec'][contains(text(),'Terbaru')]")
                driver.execute_script("arguments[0].click();", terbaru_button)
                def get_review_summary(result_set):
                    review_texts = []  # List untuk menyimpan teks review
                    id_ulasan=[]
                
                    for result in result_set:
                        articles = result.find_all('div', class_='m6QErb XiKgde')
                        for article in articles:
                            all_divs = article.find_all('div', class_='MyEned')
                            if all_divs:
                                ids = [div.get('id') for div in all_divs if div.get('id')]
                                id_ulasan.append(ids)
                                for div in all_divs:
                                    ext_data = div.find_all('span', class_='wiI7pd')  # Menemukan semua elemen <span> dengan kelas 'pan'
                                    ext_data = [span.get_text(strip=True) for span in ext_data]  # Mengambil teks dari setiap elemen <span> dan menghapus spasi
                    
                                                    # Iterasi melalui setiap teks di ext_data dan ambil kalimat pertama
                                    for text in ext_data:
                                        first_sentence = text  # Mengambil kalimat pertama sebelum titik
                                        review_texts.append(first_sentence)  # Simpan kalimat pertama ke dalam list
                
                    return review_texts,id_ulasan
                    time.sleep(5)
                    def scroll_div_until_element_found(driver, container_xpath, target_text, pause_time=2, max_scrolls=50):
                        """
                        Menggulir elemen container hingga menemukan elemen dengan teks tertentu.
                        
                        Args:
                            driver: Selenium WebDriver instance.
                            container_xpath: XPath container yang dapat digulir.
                            target_text: Teks yang dicari di dalam elemen.
                            pause_time: Waktu jeda (dalam detik) setelah setiap scroll.
                            max_scrolls: Jumlah maksimum scroll untuk mencegah loop tak terbatas.
                    
                        Returns:
                            WebElement: Elemen yang ditemukan, atau None jika tidak ditemukan.
                        """
                        scroll_count = 0
                        scrollable_div = driver.find_element(By.XPATH, container_xpath)
                        
                        while scroll_count < max_scrolls:
                            try:
                                # Cari elemen berdasarkan teks di dalam container
                                element = driver.find_element(By.XPATH, f"{container_xpath}//span[@class='rsqaWe' and text()='{target_text}']")
                                print("Elemen ditemukan!")
                                return element
                            except:
                                pass  # Jika elemen belum ditemukan, lanjutkan scroll
                                
                            # Scroll container ke bawah
                            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
                            time.sleep(pause_time)  # Tunggu konten memuat
                            
                            scroll_count += 1
                            print(f"Scroll ke-{scroll_count}")
                    
                        print("Elemen tidak ditemukan setelah menggulir container.")
                        return None
                        # Scroll container hingga menemukan elemen
                        container_xpath = "//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]"
                        element = scroll_div_until_element_found(driver, container_xpath,waktu, pause_time=2)
                        flat_data = [item for sublist in id_ulasan for item in sublist]
                        import pandas as pd

                        # Buat dataframe
                        datas = {'id_review': flat_data, 'Review': review_texts}
                        data_scrapping = pd.DataFrame(data)
                        
                        data_scrapping


                        if element:
                            print("Teks ditemukan:", element.text)
                        else:
                            print("Teks tidak ditemukan.")

                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = data_scrapping[]

                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()

                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])}
                    for i in range(new_X.shape[0])
                ]

                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            import time
            time.sleep(5)
            # Menutup driver
            driver.quit()

        if st.button("Pantai Slopeng"):
            try:
                from bs4 import BeautifulSoup
                from selenium import webdriver
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.common.exceptions import NoSuchElementException, WebDriverException
                from selenium.webdriver.chrome.service import Service
                import os
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--no-sandbox")
                
                driver = webdriver.Chrome(options=options)
                # URL dari Google Search
                url = 'https://www.google.com/maps/place/Pantai+Slopeng/@-6.8861093,113.7820433,15z/data=!4m8!3m7!1s0x2dd9ea23fabac2df:0x8550176773c06614!8m2!3d-6.8861095!4d113.792343!9m1!1b1!16s%2Fg%2F112yfwt6c?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3Dhl=id&gl=ID'
                driver.get(url)
                response = BeautifulSoup(driver.page_source, 'html.parser')
                reviews = response.find_all('div', class_='w6VYqd')
                def get_review_summary(result_set):
                    review_texts = []  # List untuk menyimpan teks review

                    for result in result_set:
                        articles = result.find_all('div', class_='m6QErb XiKgde')
                        for article in articles:
                            all_divs = article.find_all('div', class_='MyEned')
                            for div in all_divs:
                                ext_data = div.find_all('span', class_='wiI7pd')  # Menemukan semua elemen <span> dengan kelas 'pan'
                                ext_data = [span.get_text(strip=True) for span in ext_data]  # Mengambil teks dari setiap elemen <span> dan menghapus spasi

                                # Iterasi melalui setiap teks di ext_data dan ambil kalimat pertama
                                for text in ext_data:
                                    first_sentence = text  # Mengambil kalimat pertama sebelum titik
                                    review_texts.append(first_sentence)  # Simpan kalimat pertama ke dalam list

                    return review_texts
                review_texts=get_review_summary(reviews)

                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = review_texts[-5:]

                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()

                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])}
                    for i in range(new_X.shape[0])
                ]

                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            import time
            time.sleep(5)
            # Menutup driver
            driver.quit()

        if st.button("Pantai Sembilan"):
            try:
                from bs4 import BeautifulSoup
                from selenium import webdriver
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.common.exceptions import NoSuchElementException, WebDriverException
                from selenium.webdriver.chrome.service import Service
                import os
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--no-sandbox")
                
                driver = webdriver.Chrome(options=options)
                # URL dari Google Search
                url = 'https://www.google.com/maps/place/Pantai+Sembilan+Sumenep/@-7.1751703,113.919241,17z/data=!4m8!3m7!1s0x2dd759ba4659b12b:0x5818009169d7abb7!8m2!3d-7.1751703!4d113.9218159!9m1!1b1!16s%2Fg%2F11c5339dr4?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D&hl=id&gl=ID'
                driver.get(url)
                response = BeautifulSoup(driver.page_source, 'html.parser')
                reviews = response.find_all('div', class_='w6VYqd')
                def get_review_summary(result_set):
                    review_texts = []  # List untuk menyimpan teks review

                    for result in result_set:
                        articles = result.find_all('div', class_='m6QErb XiKgde')
                        for article in articles:
                            all_divs = article.find_all('div', class_='MyEned')
                            for div in all_divs:
                                ext_data = div.find_all('span', class_='wiI7pd')  # Menemukan semua elemen <span> dengan kelas 'pan'
                                ext_data = [span.get_text(strip=True) for span in ext_data]  # Mengambil teks dari setiap elemen <span> dan menghapus spasi

                                # Iterasi melalui setiap teks di ext_data dan ambil kalimat pertama
                                for text in ext_data:
                                    first_sentence = text  # Mengambil kalimat pertama sebelum titik
                                    review_texts.append(first_sentence)  # Simpan kalimat pertama ke dalam list

                    return review_texts
                review_texts=get_review_summary(reviews)

                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = review_texts[-5:]

                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()

                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])}
                    for i in range(new_X.shape[0])
                ]

                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            import time
            time.sleep(5)
            # Menutup driver
            driver.quit()

        if st.button("Air Terjun Toroan"):
            try:
                from bs4 import BeautifulSoup
                from selenium import webdriver
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.common.exceptions import NoSuchElementException, WebDriverException
                from selenium.webdriver.chrome.service import Service
                import os
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--no-sandbox")
                
                driver = webdriver.Chrome(options=options)
                # URL dari Google Search
                url = 'https://www.google.com/maps/place/Air+Terjun+Toroan/@-6.8928897,113.3097483,17z/data=!4m8!3m7!1s0x2e0518178cebfebb:0xefcf0aa128f79400!8m2!3d-6.8928897!4d113.3123232!9m1!1b1!16s%2Fg%2F11b6pkts1w?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D&hl=id&gl=ID'
                driver.get(url)
                response = BeautifulSoup(driver.page_source, 'html.parser')
                reviews = response.find_all('div', class_='w6VYqd')
                def get_review_summary(result_set):
                    review_texts = []  # List untuk menyimpan teks review

                    for result in result_set:
                        articles = result.find_all('div', class_='m6QErb XiKgde')
                        for article in articles:
                            all_divs = article.find_all('div', class_='MyEned')
                            for div in all_divs:
                                ext_data = div.find_all('span', class_='wiI7pd')  # Menemukan semua elemen <span> dengan kelas 'pan'
                                ext_data = [span.get_text(strip=True) for span in ext_data]  # Mengambil teks dari setiap elemen <span> dan menghapus spasi

                                # Iterasi melalui setiap teks di ext_data dan ambil kalimat pertama
                                for text in ext_data:
                                    first_sentence = text  # Mengambil kalimat pertama sebelum titik
                                    review_texts.append(first_sentence)  # Simpan kalimat pertama ke dalam list

                    return review_texts
                review_texts=get_review_summary(reviews)

                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = review_texts[-5:]

                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()

                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])}
                    for i in range(new_X.shape[0])
                ]

                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            import time
            time.sleep(5)
            # Menutup driver
            driver.quit()

        if st.button("Pantai Lombang "):
            try:
                from bs4 import BeautifulSoup
                from selenium import webdriver
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.common.exceptions import NoSuchElementException, WebDriverException
                from selenium.webdriver.chrome.service import Service
                import os
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--no-sandbox")
                
                driver = webdriver.Chrome(options=options)
                # URL dari Google Search
                url = 'https://www.google.com/maps/place/Pantai+Lombang/@-6.9178648,114.0599177,16z/data=!4m8!3m7!1s0x2dd9f7276ab8c685:0xe6566e3638889a6!8m2!3d-6.9155738!4d114.0586496!9m1!1b1!16s%2Fg%2F112yfp277?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D&hl=id&gl=ID'
                driver.get(url)
                response = BeautifulSoup(driver.page_source, 'html.parser')
                reviews = response.find_all('div', class_='w6VYqd')
                def get_review_summary(result_set):
                    review_texts = []  # List untuk menyimpan teks review

                    for result in result_set:
                        articles = result.find_all('div', class_='m6QErb XiKgde')
                        for article in articles:
                            all_divs = article.find_all('div', class_='MyEned')
                            for div in all_divs:
                                ext_data = div.find_all('span', class_='wiI7pd')  # Menemukan semua elemen <span> dengan kelas 'pan'
                                ext_data = [span.get_text(strip=True) for span in ext_data]  # Mengambil teks dari setiap elemen <span> dan menghapus spasi

                                # Iterasi melalui setiap teks di ext_data dan ambil kalimat pertama
                                for text in ext_data:
                                    first_sentence = text  # Mengambil kalimat pertama sebelum titik
                                    review_texts.append(first_sentence)  # Simpan kalimat pertama ke dalam list

                    return review_texts
                review_texts=get_review_summary(reviews)

                    # Mengambil 10 data pertama dari kolom 'ulasan'
                top_10_reviews = review_texts[-5:]

                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()

                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])}
                    for i in range(new_X.shape[0])
                ]

                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            import time
            time.sleep(5)
            # Menutup driver
            driver.quit()


        
          


        
