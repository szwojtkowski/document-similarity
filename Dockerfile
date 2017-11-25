FROM alpine:3.6
COPY . /opt/src/
RUN apk add --update python3 && pip3 install --trusted-host pypi.python.org -r /opt/src/requirements.txt
ENTRYPOINT ["FLASK_APP=app.py flask run"]
