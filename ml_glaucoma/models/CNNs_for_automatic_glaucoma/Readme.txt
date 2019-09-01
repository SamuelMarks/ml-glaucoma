Paper: CNNs for automatic glaucoma assessment using fundus images: an extensive validation

Folder "images": The test images need to be placed here. They need to be divided into 2 folders, "glaucoma" and "notGlaucoma" 
Folder "models": Contains json files for the models. You will have to download/train and save the weights of the models to these folders.
The models used in this work are: VGG16, VGG19, InceptionV3, ResNet50 and Xception.
The weights from this work can be downloaded from:
https://figshare.com/s/c2d31f850af14c5b5232
Folder "results": A csv file with a score for each class, "glaucoma" and "notGlaucoma" will be saved after you run the file modelEval.py.

To run the script you need to load the test images first and define the model you want to use by changing the modelName variable in the modelEval.py file.
Then the model can be run using the CLI with:
python modelEval.py
