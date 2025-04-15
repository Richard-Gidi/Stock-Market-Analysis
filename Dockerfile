# Use the official Jupyter SciPy Notebook as the base image
FROM jupyter/scipy-notebook:latest

# Set working directory inside the container
WORKDIR /home/jovyan/work

# Copy all project files into the container
COPY . .

# Expose the correct port (5000 for Flask, 8501 for Streamlit)
EXPOSE 8501  

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt  

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8
501", "--server.address=0.0.0.0"]


