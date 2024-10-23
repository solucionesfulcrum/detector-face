# Usa una imagen base de Python 3.8-slim
FROM python:3.8-slim

# Instala las dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos y la aplicación en el contenedor
COPY requirements.txt requirements.txt
COPY contar_api.py contar_api.py

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usará la aplicación Flask
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "contar_api.py"]

