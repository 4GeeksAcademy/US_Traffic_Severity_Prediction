from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
import pandas as pd

# Configura las credenciales de la API de Kaggle
api = KaggleApi()
api.authenticate()

# Nombre del dataset en Kaggle
dataset_name = 'sobhanmoosavi/us-accidents'
# Ruta donde se guardará el archivo CSV descargado
download_path = 'data/raw'
# Descarga el archivo CSV
api.dataset_download_files(dataset_name, path=download_path)
#
#  Listar los archivos descargados
downloaded_files = os.listdir(download_path)

# Ruta del archivo descargado
download_path = '/workspaces/cesargustavo-Final_Project/data/raw/us-accidents.zip'

# Directorio donde se descomprimirá el archivo
extract_path = '/workspaces/cesargustavo-Final_Project/data/raw'

# Descomprimir el archivo
with zipfile.ZipFile(download_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Listar los archivos descomprimidos
extracted_files = os.listdir(extract_path)

# Buscar el archivo CSV descomprimido (debes asegurarte de que solo haya un archivo CSV)
csv_file = [file for file in extracted_files if file.endswith('.csv')][0]

# Ruta completa al archivo CSV
csv_path = os.path.join(extract_path, csv_file)

# Leer el archivo CSV utilizando pandas
df = pd.read_csv(csv_path)

# Mostrar las primeras 10 líneas del DataFrame
print(df.head(10))