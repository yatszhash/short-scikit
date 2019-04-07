
FROM yatzhash/ubuntu-pipenv

# Pipfile: host -> container
COPY Pipfile ./
COPY Pipfile.lock ./

RUN pipenv install --system