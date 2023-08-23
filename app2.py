import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Configura las credenciales de la API de Kaggle (ruta al archivo JSON descargado)
api = KaggleApi()
api.authenticate(api_key='/workspaces/cesargustavo-Final_Project/src/kaggle.json')

# Nombre del dataset en Kaggle
dataset_name = 'sobhanmoosavi/us-accidents'

# Ruta donde se guardará el archivo CSV descargado
download_path = 'data/raw'

# Descarga el archivo CSV
api.dataset_download_files(dataset_name, path=download_path)

# Listar los archivos descargados
downloaded_files = os.listdir(download_path)

# Encuentra el archivo CSV descargado (asegúrate de que solo haya un archivo CSV)
csv_file = [file for file in downloaded_files if file.endswith('.csv')][0]

# Ruta completa al archivo CSV
csv_path = os.path.join(download_path, csv_file)

# Ahora puedes usar pandas para leer el archivo CSV y procesarlo según tus necesidades
