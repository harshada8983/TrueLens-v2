FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /home/user/app

# GitHub par naya commit aate hi cache toot'ta hai -> Space latest code se rebuild hota hai
ADD https://api.github.com/repos/harshada8983/TrueLens-v2/git/refs/heads/main /tmp/version.json
RUN git clone --depth 1 https://github.com/harshada8983/TrueLens-v2.git repo

WORKDIR /home/user/app/repo/backend
RUN pip install --no-cache-dir -r requirements.txt

# Trained model Space repo me upload hota hai (bada file, GitHub pe zaroori nahi)
COPY --chown=user truelens_model.keras config.json /home/user/app/repo/backend/model/

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
