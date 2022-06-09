FROM python:3.7
#Firstly we define our base image where we want to build our file from, as demostrated below
#We can also use the latest version of python, but this is not recommended
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
#Expose the port to be used to run the application.This is the same port that our streamlit app was running on.
EXPOSE 8501

COPY ./app.py ./app.py

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]