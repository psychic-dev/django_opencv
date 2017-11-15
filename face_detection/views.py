# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import sys

# define the path to the face detector
# FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
#     base_path=os.path.abspath(os.path.dirname(__file__)))

face_cascade = cv2.CascadeClassifier('/home/varun/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/varun/opencv/data/haarcascades/haarcascade_eye.xml')

@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        try:
            if request.FILES.get("image", None) is not None:
                # grab the uploaded image
                image = _grab_image(stream=request.FILES["image"])

            # otherwise, assume that a URL was passed in
            else:
                # grab the URL from the request
                url = request.POST.get("url", None)

                # if the URL is None, then return an error
                if url is None:
                    data["error"] = "No URL provided."
                    return JsonResponse(data)

                # load the image and convert
                image = _grab_image(url=url)

            # convert the image to grayscale, load the face cascade detector,
            # and detect faces in the image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = face_cascade.detectMultiScale(gray, 1.3, 5)

            # construct a list of bounding boxes from the detection
            rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

            # update the data dictionary with the faces detected
            data.update({"num_faces": len(rects), "faces": rects, "success": True})
        except:
            data["error"] = "Unexpected error: " + str(sys.exc_info()[0])
            return JsonResponse(data)

    # return a JSON response
    return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image