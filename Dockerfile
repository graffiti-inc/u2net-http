# Ubuntu 19.10 eoan
FROM ubuntu@sha256:9e9387cd5380eb96ace218a2f85ed25ac75563cd170f91852a73aefda6fa7834

RUN apt-get update && \
	apt-get install -y \
		libglib2.0-0=2.62.1-1 \
		libnvidia-compute-435=435.21-0ubuntu2 \
		libsm6=2:1.2.3-1 \
		libxext6=2:1.3.4-0ubuntu1 \
		libxrender1=1:0.9.10-1 \
		python3-venv=3.7.5-1 \
	&& \
	rm -rf /var/lib/apt/lists/

WORKDIR /app

# Copy U-2-Net.
COPY U-2-Net ./U-2-Net

# Copy Resnet.
COPY resnet34-333f7ec4.pth resnet34-333f7ec4.pth

# Install production dependencies.
COPY requirements.txt requirements.txt
# Don't try to uninstall existing packages, e.g., numpy
RUN python3 -m venv venv && \
	venv/bin/pip install --no-cache-dir -U pip setuptools wheel && \
	venv/bin/pip install --no-cache-dir torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
	venv/bin/python3 -m pip install --no-cache-dir -r requirements.txt
# final line is python3 -m pip because for some reason installing torch deletes the bin/pip symlink :/

# Copy local code to the container image.
COPY *.py ./

# Set default port.
ENV PORT 80

# Run the web service using gunicorn.
CMD exec venv/bin/gunicorn --bind :$PORT --workers 1 main:app
