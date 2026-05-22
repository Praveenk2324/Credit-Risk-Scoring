# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (this speeds up future builds)
COPY requirements.txt .

# 4. Install dependencies
# We explicitly force the CPU version of PyTorch to keep the container size small (~200MB vs ~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your entire project (Code, DVC pipelines, and MLflow models) into the container
COPY . .

# 6. Open port 8000 so the outside world can talk to FastAPI
EXPOSE 8000

# 7. Boot up the Uvicorn server when the container starts
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]