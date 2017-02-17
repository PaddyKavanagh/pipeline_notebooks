FROM continuumio/anaconda

# pipeline installation
RUN conda create -n jwstb7 --file http://ssb.stsci.edu/conda/jwstdp-0.7.0rc/jwstdp-0.7.7-osx-py27.0.txt
RUN source activate jwstb7