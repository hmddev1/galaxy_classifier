import os
import cv2
import numpy as np
import pandas as pd
from ZEMO import zemo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from torchvision import transforms
import tensorflow as tf
from PIL import Image

class GalaxyClassificationModels:
    def __init__(self, image_dir, zm_output_path, image_size=(200, 200), zm_order=45):
        self.image_dir = image_dir
        self.zernike_output_path = zm_output_path
        self.image_size = image_size
        self.zernike_order = zm_order
        self.zernike_repetition = 1
        
    def calculate_zernike_moments(self):
        ZBFSTR = zemo.zernike_bf(self.image_size[0], self.zernike_order, self.zernike_repetition)
        image_files = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir) if filename.endswith('.jpg')]
        zernike_moments = []
        
        for img_path in image_files:
            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, self.image_size)
            im = resized_image[:, :, 0]
            Z = np.abs(zemo.zernike_mom(np.array(im), ZBFSTR))
            zernike_moments.append(Z)
        
        df = pd.DataFrame(zernike_moments)
        df.to_csv(self.zernike_output_path, index=False)
        
        return df
    
    def load_transform_images(data_dir, target_size=(200, 200)):

        def load_images(sub_dir):
            all_images = []
            file_paths = [os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir) if filename.endswith('.jpg')]
            for img_path in file_paths:
                image = cv2.imread(img_path)
                resized_image = cv2.resize(image, target_size)
                pil_image = Image.fromarray(resized_image)
                all_images.append(pil_image)
            return all_images

        # Load images
        g_img = load_images(os.path.join(data_dir, 'galaxy'))
        ng_img = load_images(os.path.join(data_dir, 'non_galaxy'))
        all_data = g_img + ng_img

        # Create labels
        galaxies_labels = np.zeros(len(g_img))
        nongalaxy_labels = np.ones(len(ng_img))
        all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])

        # Split the data
        img_train, img_test, y_img_train, y_img_test = train_test_split(all_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
        y_train_encoded = to_categorical(y_img_train, num_classes=2)

        # Class weights
        class_weights = {0: len(all_data) / (2 * len(g_img)), 1: len(all_data) / (2 * len(ng_img))}

        # Define transforms
        train_transform = transforms.Compose([
            transforms.CenterCrop(target_size[0]),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(target_size[0], scale=(0.8, 1.0), ratio=(0.99, 1.01)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.CenterCrop(target_size[0]),
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

    def svm_zms(self):
        galaxy_zm_directory=input("Input the Galaxy's ZMs directory: ")
        nongalaxy_zm_directory=input("Input the non-galaxy's ZMs directory: ")
        
        galaxy_zms = pd.read_csv(galaxy_zm_directory)
        nongalaxy_zms = pd.read_csv(nongalaxy_zm_directory)
        
        zmg = np.array(galaxy_zms)
        zmng = np.array(nongalaxy_zms)
        all_zm_data = np.concatenate([zmg,zmng])

        galaxies_labels = np.zeros(len(zmg))
        nongalaxy_labels = np.ones(len(zmng))
        all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])

        zm_train, zm_test, y_zm_train, y_zm_test = train_test_split(all_zm_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
        class_weights = {0: len(all_zm_data) / (2 * len(zmg)), 1: len(all_zm_data) / (2 * len(zmng))}
        model = SVC(kernel='rbf', probability=True, C=1.5, gamma='scale', class_weight=class_weights)
        model.fit(zm_train, y_zm_train)
        y_pred = model.predict(zm_test)
        return y_pred
    
    def cnn_zms(self):
        galaxy_zm_directory=input("Input the Galaxy's ZMs directory: ")
        nongalaxy_zm_directory=input("Input the non-galaxy's ZMs directory: ")
        
        galaxy_zms = pd.read_csv(galaxy_zm_directory)
        nongalaxy_zms = pd.read_csv(nongalaxy_zm_directory)
        galaxy_zms.drop("Unnamed: 0", axis = 1, inplace = True)
        nongalaxy_zms.drop("Unnamed: 0", axis = 1, inplace = True)
        zmg = np.array(galaxy_zms)
        zmng = np.array(nongalaxy_zms)
        all_zm_data = np.concatenate([zmg,zmng])

        galaxies_labels = np.zeros(len(zmg))
        nongalaxy_labels = np.ones(len(zmng))
        all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])

        zm_train, zm_test, y_zm_train, y_zm_test = train_test_split(all_zm_data, all_labels, test_size=0.25, shuffle=True, random_state=104)
        class_weights = {0: len(all_zm_data) / (2 * len(zmg)), 1: len(all_zm_data) / (2 * len(zmng))}
        y_train_encoded = to_categorical(y_zm_train, num_classes=2)

        input_shape = (all_zm_data.shape[1], 1)
        b_size = 64
        e_num = 30
        
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
        m2 = MaxPooling1D(pool_size=2)(b2)
        d2 = Dropout(0.1)(m2)
        f = Flatten()(d2)
        de0 = Dense(64, activation='relu')(f)
        de1 = Dense(32, activation='relu')(de0)
        outputs = Dense(2, activation='softmax')(de1)
        
        model = Model(inputs, outputs, name="cnn_zm_45_galaxy_nonegalaxy")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(zm_train, y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=[es], class_weight=class_weights)
        y_pred = model.predict(zm_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return y_pred_labels
    
    def cnn_transformer(self, transformed_X_train, transformed_X_test, y_train_encoded, class_weights, b_size=32, e_num=10, image_size=(200, 200)):

        inputs = Input(shape=(image_size[0], image_size[1], 3))
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
        outputs = Dense(2, activation='softmax')(de1)
        
        model = Model(inputs, outputs, name="cnn_transformer_galaxy_nonegalaxy")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(np.array(transformed_X_train), y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=[es], class_weight=class_weights)
        y_pred = model.predict(np.array(transformed_X_test))
        y_pred_labels = np.argmax(y_pred, axis=1)
        return y_pred_labels
    
    def resnet50_transformer(self, transformed_X_train, transformed_X_test, y_train_encoded, class_weights, b_size=32, e_num=10):

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.image_size[0], self.image_size[1], 3))
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
        return y_pred_labels
    
    def vgg16_transformer(self, transformed_X_train, transformed_X_test, y_train_encoded, class_weights, b_size=32, e_num=10):

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.image_size[0], self.image_size[1], 3))
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
        return y_pred_labels
