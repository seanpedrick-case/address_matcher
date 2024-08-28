FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Install Lambda web adapter in case you want to run with with an AWS Lamba function URL
#COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.3 /lambda-adapter /opt/extensions/lambda-adapter

# Update apt
RUN apt-get update && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements_aws.txt .

RUN pip install --no-cache-dir torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
	pip install --no-cache-dir -r requirements_aws.txt && \
	pip install --no-cache-dir gradio==4.42.0

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Change ownership of /home/user directory
RUN chown -R user:user /home/user

# Make output folder
RUN mkdir -p /home/user/app/output && chown -R user:user /home/user/app/output && \
	mkdir -p /home/user/app/output/api && chown -R user:user /home/user/app/output/api && \
	mkdir -p /home/user/app/api && chown -R user:user /home/user/app/api && \
	chown -R user:user /home/user/app

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_SERVER_PORT=7861 \
	GRADIO_THEME=huggingface \
	AWS_STS_REGIONAL_ENDPOINT=regional \
	#GRADIO_TEMP_DIR=$HOME/tmp \
	#GRADIO_ROOT_PATH=/address-match \
	SYSTEM=spaces
 
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app
#COPY . $HOME/app

CMD ["python", "app.py"]