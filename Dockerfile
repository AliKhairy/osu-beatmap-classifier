# 1. Start from an existing image that already has Python 3.12 installed.
FROM python:3.12-slim

# 2. Set the folder inside the container where everything happens.
WORKDIR /app

# 3. Copy ONLY requirements.txt in (not the whole project yet).
COPY requirements.txt .

# 4. Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# 5. NOW copy the rest of the source code in.
COPY . .

# 6. What runs when the container starts.
# ENTRYPOINT is fixed; CMD is only the DEFAULT arguments to it. Anything passed
# after the image name replaces the CMD but keeps the ENTRYPOINT, so:
#   docker run osu-classifier                   -> python cli.py --help
#   docker run osu-classifier train-ensemble    -> python cli.py train-ensemble
# main.py is unchanged and still available for interactive use:
#   docker run -it --entrypoint python osu-classifier main.py
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]

