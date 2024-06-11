# main.py
import sys
from models import classifiers

def main():
    image_dir = 'C:/Users/HMDdev/Pictures/non'
    zm_output_path = 'C:/Users/HMDdev/Pictures/non/zernike_non.csv'
    model = classifiers.GalaxyClassificationModels(image_dir, zm_output_path)

    if len(sys.argv) < 2:
        print("Usage: python main.py <function> [<args>...]")
        sys.exit(1)

    function = sys.argv[1]

    if function == 'calculate_zernike_moments':
        df = model.calculate_zernike_moments()
        print("Zernike moments calculated and saved.")
    elif function == 'svm_zms':
        y_pred = model.svm_zms()
        print("SVM Zernike Moments prediction completed.")
        print("Predicted value:", y_pred)
    elif function == 'cnn_zms':
        y_pred = model.cnn_zms()
        print("CNN Zernike Moments prediction completed.")
        print("Predicted value:", y_pred)
    elif function == 'cnn_transformer':
        transformed_X_train, transformed_X_test, y_train_encoded, y_img_test, class_weights = model.load_transform_images()
        y_pred = model.cnn_transformer(transformed_X_train, transformed_X_test, y_train_encoded, class_weights)
        print("CNN Transformer prediction completed.")
        print("Predicted value:", y_pred)
    elif function == 'resnet50_transformer':
        transformed_X_train, transformed_X_test, y_train_encoded, class_weights = model.load_transform_images(image_dir)
        y_pred = model.resnet50_transformer(transformed_X_train, transformed_X_test, y_train_encoded, class_weights)
        print("ResNet50 Transformer prediction completed.")
        print("Predicted value:", y_pred)
    elif function == 'vgg16_transformer':
        transformed_X_train, transformed_X_test, y_train_encoded, class_weights = model.load_transform_images(image_dir)
        y_pred = model.vgg16_transformer(transformed_X_train, transformed_X_test, y_train_encoded, class_weights)
        print("VGG16 Transformer prediction completed.")
        print("Predicted value:", y_pred)
    else:
        print("Unknown function. Please use one of the following: calculate_zernike_moments, svm_zms, cnn_zms, cnn_transformer, resnet50_transformer, vgg16_transformer")

if __name__ == "__main__":
    main()
