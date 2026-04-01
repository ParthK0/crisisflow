# Base image
FROM python:3.11-slim

# Create user with UID 1000 for HF Spaces
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:7860/health', timeout=5); r.raise_for_status()"

# Run the app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
