from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Inicializar el navegador (asegúrate de tener el driver adecuado en tu PATH)
driver = webdriver.Chrome()

# URL del dataset en Kaggle
dataset_url = 'https://www.kaggle.com/account/login?phase=emailSignIn&returnUrl=%2F'

# Cargar la página del dataset
driver.get(dataset_url)


# Llenar los campos de correo y contraseña
email_field = driver.find_element(By.NAME, 'email')
password_field = driver.find_element(By.NAME, 'password')

email = "cesarseneca@hotmail.com"
password = 'holahola2'

email_field.send_keys(email)
password_field.send_keys(password)

sign_in_button = driver.find_element(By.XPATH, '//button[contains(., "Sign In")]')
sign_in_button.click()

# Esperar a que la página procese el registro
time.sleep(5)

# URL del dataset en Kaggle
dataset_url = 'https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents'

# Cargar la página del dataset
driver.get(dataset_url)

# Esperar un momento para que cargue la página
time.sleep(2)

# Encontrar y hacer clic en el botón de descarga por su texto
download_button = driver.find_element(By.XPATH, '//button[contains(., "Download (685 MB)")]')
download_button.click()

# Esperar un momento para que se inicie la descarga
time.sleep(20)  # Puedes ajustar el tiempo de espera según tu conexión y velocidad de descarga

# Cerrar el navegador
driver.quit()