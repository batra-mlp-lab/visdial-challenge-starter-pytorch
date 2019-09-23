from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.manifold import TSNE
import os
from sklearn.cluster import KMeans


model = VGG16(weights='imagenet', include_top=False)
model.summary()

def extract(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  vgg16_feature = model.predict(img_data)
  return np.array(vgg16_feature).flatten()

# print(type(vgg16_feature))
# path1 = 'val2014/COCO_val2014_000000423093.jpg'
# path2 = 'val2014/COCO_val2014_000000581632.jpg'


# # print(vgg16_feature.shape)
# # print(X_embedded.shape)
# print(X_embedded)

vgg16_feature_list = []

i = 0
for filename in os.listdir('val2014'):
  vgg16_feature_list.append(extract('val2014/' + filename))

  i += 1
  if i > 10:
    break

vgg16_feature_list_np = np.array(vgg16_feature_list)
X_embedded = TSNE(n_components=2).fit_transform(vgg16_feature_list_np)

print(X_embedded)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X_embedded)

print('*' * 20)
print(type(kmeans))
print(kmeans.labels_)
