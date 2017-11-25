FROM python:3.6.1

WORKDIR /app
ADD requirements.txt /app
RUN cd /app && pip3 install -r requirements.txt

ADD . /app

CMD FLASK_APP=app.py flask run --host=0.0.0.0
