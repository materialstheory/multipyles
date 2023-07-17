FROM ubuntu:jammy

# get essential packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    sudo \
    rsync \
    gfortran \
    libmpich-dev \
    mpich \
    git \
    libgmp-dev \
    libboost-dev \
    libfftw3-dev \
    libnfft3-dev \
    less \
    libmkl-full-dev \
    && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Add VASP v6.3.0 and the makefile
ADD vasp.6.3.0.tgz /
RUN mv /vasp.6.3.0 /vasp
COPY LDApU.F.diff /vasp/src/LDApU.F.diff
RUN cd vasp/src && patch LDApU.F LDApU.F.diff && grep -n uses_multipole_perturbation LDApU.F
COPY vasp.makefile.include /vasp/makefile.include

RUN cd /vasp \
    && make std \
    && rm -rf src/ build/ arch/ testsuite/

ENV PATH=/vasp/bin:${PATH}

# make sure openmp does not start
ENV OMP_NUM_THREADS=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# change user and group id to match host machine if options are passed accordingly
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Add info command
RUN mkdir /documentation
COPY README.md /documentation
RUN chmod a+r /documentation/README.md && echo "#!/bin/bash\ncat /documentation/README.md" > /documentation/info && chmod +x /documentation/info
ENV PATH=/documentation:${PATH}

