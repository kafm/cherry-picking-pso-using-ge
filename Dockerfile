#FROM ubuntu:latest

FROM python:3.8.9

RUN pip3 -q install pip --upgrade

WORKDIR /app

RUN pip3 install jupyter

RUN pip3 install numpy

RUN pip3 install pandas

RUN pip3 install matplotlib

RUN pip3 install scipy

RUN pip3 install benchmark_functions

#RUN pip3 install deap

RUN pip3 install numba
