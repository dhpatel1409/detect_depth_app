#start with base image with minimal linux and python 3.10 installed , slim is for smaller size 
FROM python:3.10-slim 

#setting working directory as cd in linux , creates direc and moves into it
WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

#COPY source destination , copy requirements.txt from local to working directory in container(. represents current working directory)
COPY requirements.txt .

# RUN apt-get update && apt-get install -y libgl1
#execute in container to install dependencies from requirements.txt file , do not want cache in container
RUN pip install --no-cache-dir -r requirements.txt

#copy all files to current working directory
COPY . .

#container listens on port 8000 (the same port uvicorn uses). This is mostly informational/convention
EXPOSE 8000

#same as running uvicorn app.main:app --host
CMD ["uvicorn", "app.main:app","--host", "0.0.0.0", "--port", "8000","--reload"]