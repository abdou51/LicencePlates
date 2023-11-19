from django.http import HttpRequest
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework import status
from PIL import Image
import pytesseract
import cv2
import numpy as np
import os
import uuid
from ultralytics import YOLO


model = YOLO("best.pt")

OUTPUT_DIR = "processed_images"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class LicensePlateRecognition(APIView):
    parser_classes = (FormParser, MultiPartParser)
    
    def post(self, request):
        image = request.data.get('image')

        if not image:
            return Response({'error': 'Image not provided'}, status=404)

        im1 = Image.open(image)
        results = model.predict(source=im1, save=False, conf=0.75)
        if (len(results[0].boxes) == 0):
            return Response({'error': 'No Licence Plate Found in the Provided Image'}, status=404)
        else:
            result = results[0]
        box = result.boxes[0]
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        cropped_im = im1.crop((xmin, ymin, xmax, ymax))

        cropped_im_np = np.array(cropped_im)

        gray_plate = cv2.cvtColor(cropped_im_np, cv2.COLOR_BGR2GRAY)
        blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
        _, binary_plate = cv2.threshold(blurred_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                character = binary_plate[y:y+h, x:x+w]
                characters.append(character)
        recognized_digits = []
        for character in characters:
            text = pytesseract.image_to_string(character, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            recognized_digits.append(text.strip())

        license_plate_number = ''.join(recognized_digits)

        unique_filename = str(uuid.uuid4()) + '.jpg'
        cropped_im_path = os.path.join(OUTPUT_DIR, unique_filename)
        cv2.imwrite(cropped_im_path, binary_plate)

        request_host = request.get_host()
        cropped_im_relative_path = os.path.join('/', OUTPUT_DIR, unique_filename).replace('\\', '/')
        cropped_im_full_url = f'http://{request_host}{cropped_im_relative_path}'
        response_data = {
            'cropped_im_url': cropped_im_full_url,
            'license_plate_number': license_plate_number,
            'sucess':'Licence Plate found !'
        }

        return Response(response_data, status=status.HTTP_200_OK)