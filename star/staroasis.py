from deepface import DeepFace
import numpy as np
import torch.nn.functional as F

def avg_score(score,num):
  if len(score)>=num:
    return sum(score[:num])/num
  else:
    return sum(score)/len(score)

def softmax(x):  
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
  
def base_model(image):
  hybe = DeepFace.find(img_path = image,    # the image to compare against
              db_path = "hybe",    # folder containing all the images
              model_name = 'ArcFace',
              enforce_detection = False)[0]

  sm = DeepFace.find(img_path = image,    # the image to compare against
                db_path = "sm",    # folder containing all the images
                model_name = 'ArcFace',
                enforce_detection = False)[0]

  jyp = DeepFace.find(img_path = image,    # the image to compare against
                db_path = "jyp",    # folder containing all the images
                model_name = 'ArcFace',
                enforce_detection = False)[0]

  yg = DeepFace.find(img_path = image,    # the image to compare against
                db_path = "yg",    # folder containing all the images
                model_name = 'ArcFace',
                enforce_detection = False)[0]
  lst = ['jyp','hybe','yg','sm']
  data = [avg_score(jyp['ArcFace_cosine'],3),avg_score(hybe['ArcFace_cosine'],3),avg_score(yg['ArcFace_cosine'],3),avg_score(sm['ArcFace_cosine'],3)]
  target = lst[np.argmin(data)]
  probabilities = softmax(data)[np.argmin(data)]
  identity = eval(target)['identity'][0].split('/')[-1].split('_')[0]
  return target, probabilities,identity

if __name__ == '__main__':
    t, p,i = base_model('jyp/sana.jpeg')
    print(f'당신이 {t}상일 확률은 {p}입니다!')
    print(f'특히 {i} 아티스트를 가장 닮았습니다')
