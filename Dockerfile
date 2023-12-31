FROM python:3.10.12

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv

RUN virtualenv venv -p python3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for flask
EXPOSE 5000

# Run application
ENV PYTHONPATH=/app
CMD ["python", "src/app.py"]
