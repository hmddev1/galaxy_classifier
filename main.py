# main.py
import sys
from models import classifiers

def main():

    model = classifiers.GalaxyClassificationModels()

    if len(sys.argv) < 2:
        print("Usage: python main.py <function> [<args>...]")
        sys.exit(1)

    function = sys.argv[1]

    if function == 'zm_cal':
        df = model.calculate_zernike_moments()
        print("Zernike moments calculated and saved.")
    elif function == 'seg_crop':
        model.segmentation_crop()
        print("The images are cropped.")
    elif function == 'model_i':
        y_zm_test, y_pred = model.model_I()
        print("SVM Zernike Moments prediction completed.")
        print("Actual labels:", y_zm_test)
        print("Predicted labels:", y_pred)
    elif function == 'model_ii':
        y_zm_test, y_pred = model.model_II()
        print("CNN Zernike Moments prediction completed.")
        print("Actual labels:", y_zm_test)
        print("Predicted value:", y_pred)
    elif function == 'model_iii':
        y_img_test, y_pred = model.model_III()
        print("CNN Transformer prediction completed.")
        print("Actual labels:", y_img_test)
        print("Predicted value:", y_pred)
    elif function == 'model_iv':
        y_img_test,y_pred = model.model_IV()
        print("ResNet50 Transformer prediction completed.")
        print("Actual labels:", y_img_test)
        print("Predicted value:", y_pred)
    elif function == 'model_v':
        y_img_test, y_pred = model.model_V()
        print("VGG16 Transformer prediction completed.")
        print("Actual labels:", y_img_test)
        print("Predicted value:", y_pred)
    else:
        print("Unknown function. Please use one of the following: calculate_zernike_moments, svm_zms, cnn_zms, cnn_transformer, resnet50_transformer, vgg16_transformer")

if __name__ == "__main__":
    main()
