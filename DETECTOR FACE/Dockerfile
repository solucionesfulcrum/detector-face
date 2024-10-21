# Usa una imagen base de Python 3.8.1rc1
FROM python:3.8.1rc1-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos y la aplicación en el contenedor
COPY requirements.txt requirements.txt
COPY app.py app.py

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usará la aplicación Flask
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
