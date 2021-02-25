# U^2-Net HTTP

This is an HTTP service wrapper of the model: [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net) (Qin et al, Pattern Recognition 2020)

The deploy folder contains configuration files for deployment as serverless container with Knative.

# Development

-   Clone this repository: `git clone git@github.com:graffiti-inc/u2net-http.git`
-   Go into the cloned directory: `cd u2net-http`
-   Clone the official [U^2-Net repository](https://github.com/NathanUA/U-2-Net)
-   Download the pretrained model [u2net.pth](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view)
-   Put the file inside the `U-2-Net/saved_models/u2net/` folder.

# Build from source

### Option 1 - Using Docker

After you've retrieved the U^2-Net model.

Download Resnet checkpoint

```
curl https://download.pytorch.org/models/resnet34-333f7ec4.pth -o resnet34-333f7ec4.pth
```

```bash
docker build -t graffiti-u2net-http .

docker run -p 8080:80 -e PYTHONUNBUFFERED=x graffiti-u2net-http
```

Pay attention to the version of `libnvidia-compute` in the Dockerfile. It should match what is on every host you are deploying to. Check the version by running `dpkg -l | grep libnvidia-compute`.

### Option 2 - Locally with virtualenv

```bash
virtualenv venv
venv/bin/activate
```

```bash
pip install torch==0.4.1
pip install -r requirements.txt
```

```
python main.py
```

# Test

```bash
$ curl -F "data=@test.jpg" http://localhost:8080/

> Hello U^2-Net!
```

```bash
$ curl -F "data=@test.jpg" http://localhost:8080/image -o result.png
```
