FROM python:3.6-jessie

RUN pip install keras Pillow h5py tensorflow

ADD https://github.com/iberryful/captcha_break/raw/master/fonts/Courier%20New.ttf  /courier_new.ttf

ADD https://github.com/iberryful/captcha_break/raw/master/vin.h5 /vin.h5

ADD ://github.com/iberryful/captcha_break/raw/master/vin/vin.py /vin.py 

WORKDIR /

CMD python vin.py
