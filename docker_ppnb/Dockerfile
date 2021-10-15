# RUN IN TERMINAL: 
# docker build -t gators_pp:latest -f docker_ppnb/Dockerfile .\
# gators docker image (for PP NBs)
# If getting ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE 
# REQUIREMENTS FILE, just keep re-running

# Install default environment
FROM dockerhub.paypalcorp.com/core-data-platform/ppmagics-spark230:latest
ENV LANG C.UTF-8
# RUN echo python --version

# Delete cache
#stops ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE)
RUN rm ~/.cache/pip -rf

# Install python libs
# RUN apt-get update && apt-get install -y python python-dev python-pip virtualenv libssl-dev libpq-dev git build-essential libfontconfig1 libfontconfig1-dev


# Copy gators directory
RUN pip install -U pip --no-cache
RUN pip install setuptools>=41.0.0 --no-cache
RUN pip install numpy==1.19.5 --no-cache
RUN pip install cython --no-cache
RUN pip install wheel --no-cache

COPY ./ /gators
# RUN chmod 777 /gators/
RUN pip install  /gators/. --no-deps
# RUN python /gators/setup_docker.py build_ext --inplace
# Install gators
