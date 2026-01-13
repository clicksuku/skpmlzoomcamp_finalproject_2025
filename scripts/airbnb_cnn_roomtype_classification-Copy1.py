# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3.10 (tfenv)
#     language: python
#     name: tfenv
# ---

# %% id="XgxqfHiw9O1X"
import os
import ssl
import requests
from io import BytesIO
from urllib import request

from tqdm import tqdm
from PIL import Image

import pandas as pd
import json
import numpy as np
import seaborn as sn
import pickle

from matplotlib import pyplot as plt
from io import StringIO
# %matplotlib inline

# %% id="3bzuJKLx-D2s"
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# %%
# !python --version

# %%
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models
import tensorflow_datasets as tfds

# %%
my_data_dir = '../Data_tfds' 
test_images_dir = '../test'

# %% colab={"base_uri": "https://localhost:8080/", "height": 251, "referenced_widgets": ["2336bd4ab5274331b51134b1287fa299", "8aa22a74ee0a4677b42ac7460ce51385", "9bcf1034df2b4282b25fc72e18934fa8", "22132b67ee714aeb921d3d48c29448e8", "b56a48ea12d74cb1a908eae2b4121460", "886e51913c65436db74511f9345f1b00", "67035074d8b240b1901e64b2faaec34e", "191b8585e9d14764afa388ff82a0685e", "b7bbbe358235464596a20931ee7d45f0", "faa76acaf2b44cdf8b71afd0cfee6c7f", "16c4dbb396ec48ed802397c3ba9db4db", "bc0df59cbe3344b0a9872886d9ed316d", "6fd0ff8643af452dbe06b00f9c51ba3a", "0adfd74d1c5249dc8541ea603ec28ba8", "b7b7e4307fd64eeaaf529f62be43a6ae", "df1d78e71afd492f8e4affd1e0074215", "a8e12b2f3eee40c2acc9a1a9f81f3653", "7ea40d736a1e4a9e9b9b856a8dcd44e0", "fdadcba2e483481da217862fc1fb17b8", "ccd39d3733a8408a8c32c01b233e5ccf", "b6ebd504534f40ecad09f0bf187cdca3", "06c7f9de00a043058a86b47ca7a63054", "ea2797b8fc5b43748c6b0ff03ef633d7", "d7e626ee5ad3460db02923cb26ec1028", "024e137c9a3043b5bbf81680b1275a28", "29fbf29ec1f74eb38732c3fb3b27ef5f", "a7fbe7d70b2344a087411953e6a2013e", "2519ffd3b4be4aceaec7a5503b604b1a", "5a7f54d13152416b8f42691da75c891f", "391c7ddaf3a74910a91d020af5d8bd0e", "93636f22656e4d9f8e51582f709acf44", "7f45e48de6624306aa51dfca03ab25f4", "d039863d329546c8b5d6ca1d4b60937e", "52a1db1731a24680808203dba44fb2d5", "3512e27a4a41461483e04cb75dfe86dc", "f117569da4104077bd29530b25912a3a", "77b14d2ecd804d59babdcde053cbf1e0", "e7508c67894d4e21b8081f9b165db190", "a218bf7f543b4d7591c264da20b929d6", "02d35ced5fec44c7ba56b3b42087ea85", "cae5efd02e194437b156bb33e47a2cab", "3a80a00b030a407d8d31e68617878a9d", "a2ead19f98a04daca0b9d2bd51588c0d", "0644a2b19ba44fab900d57d192bc469f", "9bc476ce9f474eee9a80730b02dccaf1", "72eab8f1412e4ce18e379a54134340f2", "700ceb9c753e43abb199b95f5779a5b7", "36f732c20bd448e6bac1c1b733d0c256", "01fc5f5b7f71401db698aaa3734beedb", "9c7ea36259614615ace15767131dc8b9", "1b5028a3ecc74ddfbd332e017d99f40f", "2a8193bd890241ee8a531136fc6f94be", "5f6ff917ff1f4057a87882a6cf5f9244", "6e6445f7a16948bc951a21821d6ec1fa", "0d62b344efaf4a49b3100ae62e863fac", "7247e7a232854378a52db8859fc33290", "7d00efec15d64e63a62443d6ba35ceff", "deeba429d3cb49adacaacdaedb1e9c70", "8af372fd7db74278bd9bf20239fecd91", "be2f399633554e89ac28ebf84c7b60df", "f424bcf3dfc54208bbbcf8e5a8f17070", "989ca42cddea4d9484222a2021c65974", "6eedbd0cae754564bfab741961d8d9ac", "4fdf2764e3ad4ceb8d8c1726f2e1bfd7", "4dffbe76d48041dab053a05d0268b98f", "bde3272244c7485ca372f45e44c37c8f", "ea57fae4cca84d6597e9ecfc58727865", "f48a10cf413d4e30a59090bcdac2f964", "f084f4136b9a439abb7018a4a3a911f4", "9e3f9c68882147f6884174722f67a11e", "b3fa4933b9394362a5f3e297be82ffa5", "a90317318743406089b57d5fb6fea279", "b6d7e03da6984a999130d2db53d922b7", "3dd6cb3ab76149dc84a838ff77659b47", "a7a8e6bac9b04c8ca152d6a5afbf48af", "4a7c4e77ffa64e38b390ef2893ddab96", "f8e3f112f3ba477b8bc08b11fb84a018", "0a0d653fe3a446fd8f1cdf1785723c12", "64a8fcd7f1234af1b5b5e242588e63d6", "5cea07a365be45e4ac3a912fc102507f", "8030e61fdd4642c7ab45fe50bace74ac", "59613d5f39e14125a919d3098a5eeb7e", "2e695903e8c6442a9c075d819a1176a3", "240f95fd2b3a494b992c1a3265650d8e", "8f29b5c09e964a78bda7e3ff5b5f2108", "683a3e82028942c08cf4486d0d8488c7", "a92cfa71e4704787b60a55e66844e5d4", "530ddd4b76a840678752db181c9b4c55", "5b260f1f2d6f4ed49b785dbe0922d106", "726786e49cdd449b9698ef308d5e094a", "218e0d80dd5f449c8bea1b7740def45e", "1c7f8457320e40d1bd721c98727ab8f5", "ca919a9c27764471824362d626146ee1", "aad75004ef8e4e67aa361e2934960828", "f0cc2abe8db44de99eeabd29e978df15", "a61aa71710954d719c997115ea1b90a3", "f9a9e5c3a9f742ba8a5efca54f36faf4", "c3cb45f4dc82453f8551e3e166d6cf2b", "6a18a68949c342abbbd75a2cefcc917b", "c4228c6785a64cd9b10465e8de537c23", "ef2c3f99367247719ab6624f3aae9136", "68c936ec0f084dd28381b1506a835128", "00e3fe54a0d74296b25866c35c6559e4", "02a2bb2c6c1149be82a5ce0cc6f91e47", "5f4b44627b1e4ca9a0c044f0afe735b8", "91bf2e559d9b4b01ab86ecf1e9747745", "d61ff8540f254f7ab8045b5b60af2daf", "01df20efbea64f4e8ba679a692d78ade", "7fbdc09e812b43bfa101c9cd79ce117c", "0240c9901caf4d9dadbf76cd0892de47", "e42439dfdea648798f34cec04a4af35d", "c2100170ec2e4b6da31463bf566937f7", "bf66f3b0de874e24a73133b516e91aee", "c06c44e2f81d4f42b9e402f7b92968d8", "1277368b9b3d4b63bcfcdfcdbfdc64e8", "7b4e42fa5dbf47a2aef2fc6788c68038", "3fc77eb72b614166924889b10ce1b3e6", "41f267adba184f2798d30731e4dd6308", "e706abf6846d40d781ecb6887060e021", "bec174c4c65349f6b60f667c0d5073ec", "196b0405285c4924bcd7f90c9eb50f1e", "8e9aab9ce38d40f7ae9b917603e59762", "bca34e0643e64757ab40cbb5b362928c", "8da336b7bfcc4d8fb8f422365a9d3e51", "a1210fb5ed434083bad7098702d8fa02", "2b6b26957a6542f6a8ca7f3b0dcd52f0", "a1bbd7ae3a5b4ba8881d55d7fe14ceae", "95d253bc032a431590abaf9f38fe4f69", "a2b16ca667094ac5ad16825021d68828", "712a21fc43a446b8ad0d575c8abede88", "c3308cb948df43ada1b63c09bd6b703b", "285b91276e1749c5a34e57242ddbecc7"]} id="CiwuazA1Qykt" outputId="7ef9aeb0-040e-4dfc-f09f-a896ec8ee58d"
(ds_train, ds_test), ds_info  = tfds.load('places365_small', split=['train','test'], data_dir=my_data_dir,
                                          shuffle_files=True, with_info=True, as_supervised=True, download=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="d8sGcBY7ReMv" outputId="5fcd8757-6a35-4384-ab9d-4f47927e60f6"
len(ds_train), len(ds_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="KWhTjjVNb7Zi" outputId="6720e9af-4d42-440d-902b-b839a6bd9553"
label_names = ds_info.features["label"].names
print("Number of classes:", len(label_names))
print(label_names[:10])

# %% id="BM2IUntndGSn"
from keras.preprocessing import image
IMG_SIZE = 224

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0).astype("float32")
    return x



# %% id="en2gcwsac4Fa"
for filename in os.listdir(test_images_dir):
  full_path = os.path.join(test_images_dir, filename)
  result = predict_image(full_path)

