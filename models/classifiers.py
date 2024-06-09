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
        self.zernike_csv_output_path = zm_output_path
        self.image_size = image_size
        self.zernike_order = zm_order
        self.zernike_repetition = 1
        self.zernike_moments = self.calculate_zernike_moments()
        
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
        df.to_csv(self.zernike_csv_output_path, index=False)
        
        return df
    
    def load_data(self, zernike_data):
        galaxies_labels = np.zeros(len(zernike_data))
        nongalaxy_labels = np.ones(len(zernike_data))
        all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])
        X_train, X_test, y_train, y_test = train_test_split(zernike_data, all_labels, test_size=0.25, shuffle=True, random_state=None)
        return X_train, X_test, y_train, y_test
    
    def svm_with_zms(self):
        zernike_data = np.array(self.zernike_moments)
        X_train, X_test, y_train, y_test = self.load_data(zernike_data)
        class_weights = {0: len(zernike_data) / (2 * len(X_train)), 1: len(zernike_data) / (2 * len(X_train))}
        model = SVC(kernel='rbf', probability=True, C=1.5, gamma='scale', class_weight=class_weights)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred
    
    def cnn_with_zms(self):
        zernike_data = np.expand_dims(np.array(self.zernike_moments), axis=2)
        X_train, X_test, y_train, y_test = self.load_data(zernike_data)
        y_train_encoded = to_categorical(y_train, num_classes=2)
        input_shape = (zernike_data.shape[1], 1)
        batch_size = 64
        num_epochs = 30
        
        inputs = Input(shape=input_shape)
        x = Conv1D(256, kernel_size=3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.1)(x)
        x = Conv1D(128, kernel_size=3, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.1)(x)
        x = Conv1D(64, kernel_size=3, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x)
        
        model = Model(inputs, outputs, name="cnn_zm_45_galaxy_nonegalaxy")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(X_train, y_train_encoded, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[es])
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return y_pred_labels
    
    def cnn_with_transformer(self, data_dir, target_size=(200, 200)):
        def load_images(data_dir, target_size):
            all_images = []
            file_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.jpg')]
            for img_path in file_paths:
                image = cv2.imread(img_path)
                resized_image = cv2.resize(image, target_size)
                resized_image = (resized_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(resized_image)
                all_images.append(pil_image)
            return all_images
        
        g_img = load_images(os.path.join(data_dir, 'galaxy'), target_size)
        ng_img = load_images(os.path.join(data_dir, 'non_galaxy'), target_size)
        all_data = g_img + ng_img
        
        galaxies_labels = np.zeros(len(g_img))
        nongalaxy_labels = np.ones(len(ng_img))
        all_labels = np.concatenate([galaxies_labels, nongalaxy_labels])
        X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.25, shuffle=True, random_state=None)
        y_train_encoded = to_categorical(y_train, num_classes=2)
        class_weights = {0: len(all_data) / (2 * len(g_img)), 1: len(all_data) / (2 * len(ng_img))}
        
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
        
        transformed_X_train = [np.transpose(train_transform(img), (1, 2, 0)) for img in X_train]
        transformed_X_test = [np.transpose(test_transform(img), (1, 2, 0)) for img in X_test]
        
        inputs = Input(shape=(target_size[0], target_size[1], 3))
        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same")(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.1)(x)
        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x)
        
        model = Model(inputs, outputs, name="cnn_transformer_galaxy_nonegalaxy")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(np.array(transformed_X_train), y_train_encoded, batch_size=64, epochs=30, validation_split=0.1, callbacks=[es], class_weight=class_weights)
        y_pred = model.predict(np.array(transformed_X_test))
        y_pred_labels = np.argmax(y_pred, axis=1)
        return y_pred_labels
    
    def resnet50_with_transformer(self, transformed_X_train, transformed_X_test, y_train_encoded, class_weights, b_size=32, e_num=10):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.image_size[0], self.image_size[1], 3))
        for layer in base_model.layers:
            layer.trainable = False
        
        x = Flatten()(base_model.output)
        x = Dense(64, activation='relu')(x)
        output = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(np.array(transformed_X_train), y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=[es], class_weight=class_weights)
        y_pred = model.predict(np.array(transformed_X_test))
        y_pred_labels = np.argmax(y_pred, axis=1)
        return y_pred_labels
    
    def vgg16_with_transformer(self, transformed_X_train, transformed_X_test, y_train_encoded, class_weights, b_size=32, e_num=10):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.image_size[0], self.image_size[1], 3))
        for layer in base_model.layers:
            layer.trainable = False
        
        x = Flatten()(base_model.output)
        x = Dense(64, activation='relu')(x)
        output = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        model.fit(np.array(transformed_X_train), y_train_encoded, batch_size=b_size, epochs=e_num, validation_split=0.1, callbacks=[es], class_weight=class_weights)
        y_pred = model.predict(np.array(transformed_X_test))
        y_pred_labels = np.argmax(y_pred, axis=1)
        return y_pred_labels
