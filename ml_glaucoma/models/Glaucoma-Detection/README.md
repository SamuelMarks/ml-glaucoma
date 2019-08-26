### Data organization
To run this model a simple data organization is needed:  
Note that in CNN.py the lines train_data_dir, validation_data_dir map to the images path.  
Hence you either need to: 
* Change the path of those lines above to your own data path. 
  
OR

* Add a directory called "data" to the "Glaucoma-Detection" directory. Inside of that "data" directory add 2 directories called "train", and "validation". 
Inside of each add 2 directories called "glaucoma", and "not_glaucoma".  

At the end, the lines above should map to the "train", "validation" directories (though, if you're following the second part of adding directories you don't have to worry about changing the path). 

### References:
https://github.com/kesaroid/Glaucoma-Detection  
"Kesar T. N, T. C Manjunath, 'Diagnosis & detection of eye diseases using Deep Convolutional Neural Networks & Raspberry Pi', Second IEEE International Conference on Green Computing & Internet of Things (IOT), ICGCIoT, IEEE ISBN: 978-1-5386-5657-0"

