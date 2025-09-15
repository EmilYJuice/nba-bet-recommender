FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Cloud Run uses PORT environment variable)
EXPOSE 8080

# Run the Streamlit application
CMD streamlit run app/streamlit_app_v2.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true