import joblib
from mtcnn import MTCNN
import cv2
import os
import numpy as np
import timm
from torchvision import transforms as T
from PIL import Image

def isolate_face(image_path):
    print(image_path)
    name = image_path.split('.')[0]
    save_path_with_idx = f'{name}_isolate.jpg'
    if os.path.isfile(save_path_with_idx):
      pass
    # Load the image
    ff = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
    # Create a face detector using MTCNN
    detector = MTCNN()
    if image.shape[-1]>3:
        image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    # Detect faces in the image
    faces = detector.detect_faces(image)
    if len(faces) ==1:
        for i, face in enumerate(faces):
          # Get the coordinates of the bounding box for the detected face
          x, y, w, h = face['box']

          # Crop the face area from the original image
          cropped_face = image[y:y+h, x:x+w]

          # Save the cropped face as a new image
          name = image_path.split('.')[0]
          save_path_with_idx = f'{name}_isolate.jpg'
          result = cv2.imwrite(save_path_with_idx,cropped_face)
    else:
        os.remove(image_path)
        raise Exception(f"No face or too many face in {image_path}. Please use another image!")
    return result, save_path_with_idx

if __name__ == '__main__':
    labels_lut = ['irin',
 'yunna',
 'minji',
 'rose',
 'hani',
 'haiin',
 'suji',
 'janny',
 'sulyoon',
 'daniel',
 'harin',
 'yu-in-na',
 'sohi',
 'nayeon',
 'sana',
 'carina',
 'zzuwi',
 'winter',
 'sandara',
 'tayeon']
    target_dict= ['sm','sm','hybe','yg','hybe','hybe','jyp','yg','jyp','hybe','hybe','yg','jyp','jyp','jyp','sm','jyp','sm','yg','sm']
    result, save_path = isolate_face('./jeni.jpg')
    my_model = timm.create_model('mobilenetv3_large_100', pretrained=True)
    my_model.eval()
    if result:
        img = cv2.imread(save_path)
        os.remove(save_path)
    else:
        raise Exception(f"전처리 이후에 이미지가 저장되지 않은 오류임. path다시 확인하기")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    VALID_TRANSFORM = T.Compose([
            T.Resize(256),
            T.ToTensor(),
        ])
    img=VALID_TRANSFORM(img)
    img = img.unsqueeze(0)
    features= my_model(img)
    # Load the best parameters from the .pkl file
    best_params_loaded = joblib.load('final.pkl')
    i= best_params_loaded.predict(features.detach().numpy())
    p = best_params_loaded.predict_proba(features.detach().numpy())
    t = target_dict[i.item()]
    print(f'당신이 {t}상일 확률은 {round(p[0][np.argmax(p)],2)}%입니다!')
    print(f'특히 {labels_lut[i.item()]} 아티스트를 가장 닮았습니다')
    