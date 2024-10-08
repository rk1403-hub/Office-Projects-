import re
from PIL import Image
import pytesseract

class ImageTextProcessor:
    def __init__(self, image_path):
        self.image_path = image_path

    def preprocess_image(self):
        # Open the image
        img = Image.open(self.image_path).convert('LA')

        # Define the new width and height
        new_width = int(img.width * 1.75)
        new_height = int(img.height * 1.75)

        # Resize the image
        resized_img = img.resize((new_width, new_height))

        return resized_img

    def perform_ocr(self, img):
        # Perform OCR on the resized image
        result = pytesseract.image_to_string(img, config='--psm 3')
        print('OCR Result:')
        print(result)  # Print OCR result
        return result

    def extract_date_time(self, result):
        # Define regular expressions for matching time and date patterns
        date_pattern = r'\d{1,2}[/\.-]\d{1,2}[/\.-]\d{4}'
        time_pattern = r'\d{1,2}:\d{2}(?:\s?[APap][Mm])?'

        # Extract date and time using regex
        date_match = re.search(date_pattern, result)
        time_match = re.search(time_pattern, result)

        extracted_date = date_match.group() if date_match else None
        extracted_time = time_match.group() if time_match else None

        return extracted_date, extracted_time

    def process_image_and_extract_datetime(self):
        resized_img = self.preprocess_image()
        ocr_result = self.perform_ocr(resized_img)
        extracted_date, extracted_time = self.extract_date_time(ocr_result)

        # Check for specific extracted text and print the corresponding message
        if "You're up to date" in ocr_result or \
           "Your device meets your organization's security policies." in ocr_result:
            print('The current software is up to date')

        if extracted_time:
            print('Extracted Time:', extracted_time)

        if extracted_date:
            print('Extracted Date:', extracted_date)
        return extracted_date, extracted_time
# Usage
if __name__ == "__main__":
    image_path = '/Users/prajwalrk/Desktop/dont_run/66145_66145 - Windows Update (Compliant).png'
    text_processor = ImageTextProcessor(image_path)

    extracted_date, extracted_time = text_processor.process_image_and_extract_datetime()