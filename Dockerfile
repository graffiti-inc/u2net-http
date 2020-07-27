FROM python:3.8.5-buster

WORKDIR /app

# Copy U-2-Net.
COPY U-2-Net ./U-2-Net

# Copy Resnet.
COPY resnet34-333f7ec4.pth resnet34-333f7ec4.pth

# Install production dependencies.
COPY requirements.txt requirements.txt
# Don't try to uninstall existing packages, e.g., numpy
RUN pip install --ignore-installed --no-cache-dir -r requirements.txt

# Copy local code to the container image.
COPY *.py ./

# Set default port.
ENV PORT 80

# Run the web service using gunicorn.
CMD exec gunicorn --bind :$PORT --workers 1 main:app
