import os
import cv2
from PIL import Image
from skimage.measure import label
from skimage.transform import resize
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb
import numpy as np
import pandas as pd
from ZEMO import zemo
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout ,BatchNormalization, MaxPooling1D, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from torchvision import transforms
import tensorflow as tf
from PIL import Image

class GalaxyClassificationModels:
    def __init__(self):
        self.num_class=int(input('Tell me the number of classes: '))
        if self.num_class==2:
            self.galaxy_img_directory = input('Input the galaxy image directory: ')
            self.nongalaxy_img_directory = input('Input the non-galaxy image directory: ')
        elif self.num_class==3:
            self.S_img_directory = input('Input the S image directory: ')
            self.E_img_directory = input('Input the E image directory: ')
            self.O_img_directory = input('Input the O image directory: ')
        else:
            print('Error...')
        self.target_size = int(input('Input the target size: '))
    
    def segmentation_crop(self):
        def cheak_1(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU, cv2.THRESH_TRUNC)
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, aa = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)
            x,y=np.where(aa==0)
            im=thresh.copy()*0
            im[x,y]=1
            im1 = label(im)
            f=im1[212,212]
            x,y=np.where(im1==f)
            imd=im1.copy()*0
            imd[x,y]=1
            return x,y,imd


        def cheak_2(img):
            segments = slic(img,n_segments=25,compactness=10)
            ss=label2rgb(segments,img,kind = 'avg')
            aa=ss[:,:,2]
            f=aa[212,212]
            x,y=np.where(aa==f)
            dd=aa.copy()*0
            dd[x,y]=1
            return x,y,dd


        def cheak_3(img,x,y):
                x1=np.min(x)-20
                x2=np.max(x)+20
                y1=np.min(y)-20
                y2=np.max(y)+20
                imn=img[x1:x2,y1:y2]
                newsize=(x2-x1,x2-x1)
                resized_im =resize(imn,newsize)
                xx = x1 + (x2 - x1) / 2 - 212
                yy = y1 + (y2 - y1) / 2 - 212
                return xx, yy,resized_im
        
        image_dir = [self.S_img_directory,self.E_img_directory,self.O_img_directory]
        class_name = ['Spiral', 'Elliptical', 'Odd_objects']

        for c in range(len(image_dir)):
                
            image_files = [os.path.join(image_dir[c], filename) for filename in os.listdir(image_dir[c]) if filename.endswith('.jpg')]
            cropped_imgs = []
            cropped_image_count = 0
            for im in image_files:
                xx=100
                yy=100
                img = cv2.imread(im)
                x, y, imd = cheak_1(img)  

                if np.min(x)>20 and np.min(y)>20 and np.max(y)<212*2-20 and np.max(x)<212*2-20:
                    xx,yy,resized_im=cheak_3(img,x,y)
                    if (np.abs(xx) >= 30) and (np.abs(yy) >= 30):
                        x, y, dd = cheak_2(img)
                        xx,yy,resized_im=cheak_3(img,x,y)
                else: 
                    x, y, dd = cheak_2(img) 
                    if np.min(x)>20 and np.min(y)>20 and np.max(y)<212*2-20 and np.max(x)<212*2-20:
                        xx,yy,resized_im=cheak_3(img,x,y)      
                        # print(xx,yy)
                if (np.abs(xx) < 30) and (np.abs(yy) < 30):
                        ms = np.shape(resized_im)
                        if ms[0] < 150:
                            crop_img = np.array(resize(resized_im, (150, 150)))
                        elif (ms[0] < 250) and (ms[0] >= 150):
                            crop_img = np.array(resize(resized_im, (250, 250)))
                        elif (ms[0] < 350) and (ms[0] >= 250):
                            crop_img = np.array(resize(resized_im, (350, 350)))
                        elif ms[0] >= 350:
                            crop_img = np.array(resize(resized_im, (450, 450)))
                                                                            
                        arr_img = np.array(crop_img)
                        cropped_imgs.append(arr_img)

                        cropped_image_count += 1
            
            output_dir=f'{image_dir[c]}/{class_name[c]}'
            os.makedirs(output_dir, exist_ok=True)
            for idx, cropped_img in enumerate(cropped_imgs):
                result = cv2.normalize(cropped_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                output_path = os.path.join(output_dir, f'cropped_image_{idx}.jpg')
                cv2.imwrite(output_path, result)


    def calculate_zernike_moments(self):
        zernike_order = int(input('Input the Zernike order: '))
        
        ZBFSTR = zemo.zernike_bf(self.target_size, zernike_order, 1)

        if self.num_class==2:
            image_dir = [self.galaxy_img_directory, self.nongalaxy_img_directory]
            class_name = ['Galaxy', 'Non_Galaxy']
        elif self.num_class==3:
            image_dir = [self.S_img_directory+'/Spiral',self.E_img_directory+'/Elliptical',self.O_img_directory+'/Odd_objects']
            class_name = ['Spiral', 'Elliptical', 'Odd_objects']
        
        # zernike_output_path = input('Input the directory to save the ZMs: ')

        for i in range(len(image_dir)):
            image_files = [os.path.join(image_dir[i], filename) for filename in os.listdir(image_dir[i]) if filename.endswith('.jpg')]
        
            zernike_moments = []
        
            for img_path in image_files:
                image = cv2.imread(img_path)
                resized_image = cv2.resize(image, (self.target_size,self.target_size))
                im = resized_image[:, :, 0]
                Z = np.abs(zemo.zernike_mom(np.array(im), ZBFSTR))
                zernike_moments.append(Z)
            
            df = pd.DataFrame(zernike_moments)

            df.to_csv(image_dir[i] + f'/{class_name[i]}_zm.csv', index=False)
        
        return df
    
    def model_I(self):

        if self.num_class==2:
            zm_directory=input("Input the ZMs directory: ")

            galaxy_zms = pd.read_csv(zm_directory + '/Galaxy_zm.csv')
            nongalaxy_zms = pd.read_csv(zm_directory + '/Non_Galaxy_zm.csv')
            zmg = np.array(galaxy_zms)
            zmng = np.array(nongalaxy_zms)
            all_zm_data = np.concatenate([zmg,zmng])

            galaxies_labels = np.zeros(len(zmg))
            nongalaxy_labels = np.ones(len(zmng))
            all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])
            class_weights = {0: len(all_zm_data) / (2 * len(zmg)), 1: len(all_zm_data) / (2 * len(zmng))}

        elif self.num_class==3:
            # zm_directory = input("Input ZMs directory: ")
            image_dir = [self.S_img_directory+'/Spiral',self.E_img_directory+'/Elliptical',self.O_img_directory+'/Odd_objects']
            spiral_data = pd.read_csv(image_dir[0] + '/Spiral_zm.csv')
            elliptical_data = pd.read_csv(image_dir[1] + '/Elliptical_zm.csv')
            odd_data = pd.read_csv(image_dir[2] + '/Odd_objects_zm.csv')
            all_zm_data = np.concatenate([spiral_data, elliptical_data, odd_data])

            spiral_label = [0] * len(spiral_data)
            elliptical_label = [1] * len(elliptical_data)
            odd_label = [2] * len(odd_data)
            all_labels = np.concatenate([spiral_label, elliptical_label, odd_label])
            class_weights = {0: len(all_zm_data) / (3*len(spiral_data)), 1: len(all_zm_data) / (3*len(elliptical_data)), 2: len(all_zm_data) / (3*len(odd_data))}

        zm_train, zm_test, y_zm_train, y_zm_test = train_test_split(all_zm_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
        model = SVC(kernel='rbf', probability=True, C=1.5, gamma='scale', class_weight=class_weights)
        model.fit(zm_train, y_zm_train)
        y_pred = model.predict(zm_test)
        acc = metrics.accuracy_score(y_zm_test, y_pred)
        print('The Acuuracy is: ', acc)

        return y_zm_test, y_pred
    
    def model_II(self):

        if self.num_class==2:
            zm_directory=input("Input the ZMs directory: ")

            galaxy_zms = pd.read_csv(zm_directory + '/Galaxy_zm.csv')
            nongalaxy_zms = pd.read_csv(zm_directory + '/Non_Galaxy_zm.csv')
            zmg = np.array(galaxy_zms)
            zmng = np.array(nongalaxy_zms)
            all_zm_data = np.concatenate([zmg,zmng])

            galaxies_labels = np.zeros(len(zmg))
            nongalaxy_labels = np.ones(len(zmng))
            all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])
            class_weights = {0: len(all_zm_data) / (2 * len(zmg)), 1: len(all_zm_data) / (2 * len(zmng))}
            zm_train, zm_test, y_zm_train, y_zm_test = train_test_split(all_zm_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
            y_train_encoded = to_categorical(y_zm_train, num_classes=2)
        
        elif self.num_class==3:
            # zm_directory = input("Input ZMs directory: ")
            image_dir = [self.S_img_directory+'/Spiral',self.E_img_directory+'/Elliptical',self.O_img_directory+'/Odd_objects']
            spiral_data = pd.read_csv(image_dir[0] + '/Spiral_zm.csv')
            elliptical_data = pd.read_csv(image_dir[1] + '/Elliptical_zm.csv')
            odd_data = pd.read_csv(image_dir[2] + '/Odd_objects_zm.csv')
            all_zm_data = np.concatenate([spiral_data, elliptical_data, odd_data])

            spiral_label = [0] * len(spiral_data)
            elliptical_label = [1] * len(elliptical_data)
            odd_label = [2] * len(odd_data)
            all_labels = np.concatenate([spiral_label, elliptical_label, odd_label])
            class_weights = {0: len(all_zm_data) / (3*len(spiral_data)), 1: len(all_zm_data) / (3*len(elliptical_data)), 2: len(all_zm_data) / (3*len(odd_data))}
            zm_train, zm_test, y_zm_train, y_zm_test = train_test_split(all_zm_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
            y_train_encoded = to_categorical(y_zm_train, num_classes=3)

        b_size=int(input('Innput the size of batches: '))
        e_num=int(input('Input the number of epochs: '))

        input_shape = (all_zm_data.shape[1], 1)
        
        inputs = Input(shape=input_shape)
        c0 = Conv1D(256, kernel_size=3, strides=2, padding="same")(inputs)
        b0 = BatchNormalization()(c0)
        m0 = MaxPooling1D(pool_size=2)(b0)
        d0 = Dropout(0.1)(m0)
        c1 = Conv1D(128, kernel_size=3, strides=2, padding="same")(d0)
        b1 = BatchNormalization()(c1)
        m1 = MaxPooling1D(pool_size=2)(b1)
        d1 = Dropout(0.1)(m1)
        c2 = Conv1D(64, kernel_size=3, strides=2, padding="same")(d1)
        b2 = BatchNormalization()(c2)
        # m2 = MaxPooling1D(pool_size=2)(b2)
        d2 = Dropout(0.1)(b2)
        f = Flatten()(d2)
        de0 = Dense(64, activation='relu')(f)
        de1 = Dense(32, activation='relu')(de0)
        outputs = Dense(self.num_class, activation='softmax')(de1)
        
        model = Model(inputs, outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(zm_train, y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=es, class_weight=class_weights)
        y_pred = model.predict(zm_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        acc = metrics.accuracy_score(y_zm_test, y_pred_labels)
        print('The Acuuracy is: ', acc)

        return y_zm_test, y_pred_labels
    

    def load_transform_images(self):

        def load_images(sub_dir, target_size):

            all_images = []
            file_paths = [os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir) if filename.endswith('.jpg')]
            for img_path in file_paths:
                image = cv2.imread(img_path)
                resized_image = cv2.resize(image, target_size)
                resized_images = (resized_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(resized_images)
                all_images.append(pil_image)
            return all_images

    
        if self.num_class==2:
                
            g_img = load_images(self.galaxy_img_directory, target_size=(self.target_size,self.target_size))
            ng_img = load_images(self.nongalaxy_img_directory, target_size=(self.target_size,self.target_size))
            all_data = g_img + ng_img

            galaxies_labels = np.zeros(len(g_img))
            nongalaxy_labels = np.ones(len(ng_img))
            all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])
            img_train, img_test, y_img_train, y_img_test = train_test_split(all_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
            y_train_encoded = to_categorical(y_img_train, num_classes=2)
            class_weights = {0: len(all_data) / (2 * len(g_img)), 1: len(all_data) / (2 * len(ng_img))}

        elif self.num_class==3:
            s_img = load_images(self.S_img_directory+'/Spiral', target_size=(self.target_size,self.target_size))
            e_img = load_images(self.E_img_directory+'/Elliptical', target_size=(self.target_size,self.target_size))
            o_img = load_images(self.O_img_directory+'/Odd_objects', target_size=(self.target_size,self.target_size))
            all_data = s_img + e_img + o_img

            spiral_label = [0] * len(s_img)
            elliptical_label = [1] * len(e_img)
            odd_label = [2] * len(o_img)
            all_labels = np.concatenate([spiral_label, elliptical_label, odd_label])

            img_train, img_test, y_img_train, y_img_test = train_test_split(all_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
            y_train_encoded = to_categorical(y_img_train, num_classes=3)
            
            class_weights = {0: len(all_data) / (3*len(s_img)), 1: len(all_data) / (3*len(e_img)), 2: len(all_data) / (3*len(o_img))}

        # Define transforms
        train_transform = transforms.Compose([
            transforms.CenterCrop(self.target_size),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(self.target_size, scale=(0.8, 1.0), ratio=(0.99, 1.01)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.CenterCrop(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Apply transforms
        transformed_X_train=[]
        for i in range(len(img_train)):
            transformed_train_images = train_transform(img_train[i])
            new_image = np.transpose(transformed_train_images, (1, 2, 0))
            transformed_X_train.append(new_image)

        transformed_X_test=[]
        for j in range(len(img_test)):
            transformed_test_images = test_transform(img_test[j])
            new_images = np.transpose(transformed_test_images, (1, 2, 0))
            transformed_X_test.append(new_images)

        return transformed_X_train, transformed_X_test, y_train_encoded, y_img_test, class_weights

    def model_III(self):

        transformed_X_train, transformed_X_test, y_train_encoded, y_img_test, class_weights = GalaxyClassificationModels.load_transform_images(self)

        b_size=int(input('Innput the size of batches: '))
        e_num=int(input('Input the number of epochs: '))

        inputs = Input(shape=(self.target_size, self.target_size, 3))
        c0 = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding="same")(inputs)
        b0 = BatchNormalization()(c0)
        m0 = MaxPooling2D(pool_size=(2, 2))(b0)
        c1 = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding="same")(m0)
        b1 = BatchNormalization()(c1)
        m1 = MaxPooling2D(pool_size=(2, 2))(b1)
        c2 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same")(m1)
        b2 = BatchNormalization()(c2)
        m2 = MaxPooling2D(pool_size=(2, 2))(b2)
        f = Flatten()(m2)
        de0 = Dense(64, activation='relu')(f)
        de1 = Dense(32, activation='relu')(de0)
        outputs = Dense(self.num_class, activation='softmax')(de1)
        
        model = Model(inputs, outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(np.array(transformed_X_train), y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=es, class_weight=class_weights)
        y_pred = model.predict(np.array(transformed_X_test))
        y_pred_labels = np.argmax(y_pred, axis=1)
        acc = metrics.accuracy_score(y_img_test, y_pred_labels)
        print('The Acuuracy is: ', acc)

        return y_img_test, y_pred_labels
    
    def model_IV(self):

        transformed_X_train, transformed_X_test, y_train_encoded, y_img_test, class_weights = GalaxyClassificationModels.load_transform_images(self)
        b_size=int(input('Innput the size of batches: '))
        e_num=int(input('Input the number of epochs: '))

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.target_size, self.target_size, 3))
        for layer in base_model.layers:
            layer.trainable = False
        
        x0 = Flatten()(base_model.output)
        x1 = Dense(64, activation='relu')(x0)
        output = Dense(self.num_class, activation='softmax')(x1)
        
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(np.array(transformed_X_train), y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=es, class_weight=class_weights)
        y_pred = model.predict(np.array(transformed_X_test))
        y_pred_labels = np.argmax(y_pred, axis=1)

        acc = metrics.accuracy_score(y_img_test, y_pred_labels)
        print('The Acuuracy is: ', acc)

        return y_img_test, y_pred_labels
    
    def model_V(self):
        transformed_X_train, transformed_X_test, y_train_encoded, y_img_test, class_weights = GalaxyClassificationModels.load_transform_images(self)
        b_size=int(input('Innput the size of batches: '))
        e_num=int(input('Input the number of epochs: '))

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.target_size, self.target_size, 3))
        for layer in base_model.layers:
            layer.trainable = False
        
        x0 = Flatten()(base_model.output)
        x1 = Dense(64, activation='relu')(x0)
        output = Dense(2, activation='softmax')(x1)
        
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(np.array(transformed_X_train), y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=[es], class_weight=class_weights)
        y_pred = model.predict(np.array(transformed_X_test))
        y_pred_labels = np.argmax(y_pred, axis=1)
        acc = metrics.accuracy_score(y_img_test, y_pred_labels)
        print('The Acuuracy is: ', acc)

        return y_img_test, y_pred_labels
