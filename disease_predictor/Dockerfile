ARG PYTHON_VERSION=3.14.2

FROM python:${PYTHON_VERSION}

EXPOSE 5100

WORKDIR /usr/project-it3160/web

COPY . .

RUN python -m pip install --no-cache-dir -r required_modules.txt

CMD ["python", "-m", "server.server"]