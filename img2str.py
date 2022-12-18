# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 05:49:20 2022

@author: Hina
"""

# -*- coding: utf-8 -*-
import base64


def img2str(file, functionName):
    img = open(file, 'rb')
    content = '{} = {}\n'.format(functionName, base64.b64encode(img.read()))
    img.close()

    with open('img2str.py', 'a') as f:
        f.write(content)


if __name__ == '__main__':
    img2str('sad_panda.jpg', 'explode')