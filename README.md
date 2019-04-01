# Flower-Image-Classifier

Flower Image Classifier application with deep-learning models that can run from the command line. Pytorch models were built and trained based on the following architectures: VGG16, AlexNet, DenseNet 161, Inception v3, and ResNet 18.

Run from the command line. For example, type: python predict.py image_05730.jpg fclassifier_inception_cp.pth --gpu

This will return the top 5 classes (default) of the flower image using the inception-based architecure trained model.

You can use other .pth files in the same directory containing best trained weights and biasis for each of the trained architectures listed above.

You can also train a model on your own using different hyperparameters for the chosen neural network architecture and save the parameters into a .pth file for use by predict script.

Cheers,
Besher
