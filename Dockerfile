FROM heroku/miniconda

# Grab requirements.txt.
#ADD ./requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr ./requirements.txt


#CMD gunicorn --bind 0.0.0.0:$PORT wsgi

