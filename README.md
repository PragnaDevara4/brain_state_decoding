The project has 3 scripts:
1.(I) For the given brain readings, generate images out of them and classify to which brain state(awake,slow_updown,MA) they belong to using CNN
2.(II) Take the binary images of the readings, add noise into them and denoise, all this by using HNN and then after denoising classify them using CNN
3.(III) Do not denoise and directly classify the images using CNN

CNN_only_1_copy.py
This script is for the (I) task as mentioned above.
For different combination of noise percent and image sizes, finding classification accuracy and confusion matrix
Methods:
1.generate_binary_image: Takes a set of readings, and min-max scaling is done on this subseries(of len 25ms i.e 250 readings of a row). Intially an image(matrix) of given size*size is created with all -1's.
each value in subseries is normalised, and scaled according to size of image(y_coordinate), x_coordinate is its position in the subseries, scaled to the size of image and plotted as 1.
2.plot_sample_images : Used to plot images 
3.prepare_training_and_testing_dataCNN : split the images into training(10%), validation(10%), testing(80%)[NOTE: HNN requires very less training data, and is the reason for taking 10% split]
4.train_and_evaluate_model : train the CNN model to classify the 3 brain states of all the images.
5.no_artifact : global_min, global_max values of all states together is taken(used for min-max scaling), and the readigs are divided into 250 window length and these are sent to generate_binary_image function

artifact_and_HNN.py
This script is for the (II) task as mentioned above.
For different combination of noise percent and image sizes, and later after gets denoised and finding classification accuracy and confusion matrix
Methods:
1.insert_artifact : according to the given size of image and error(noise) percent, the no. of pixel values to change are calculated. Those many no. of pixels are randomly selected and their values are flipped
i.e, -1 to 1 or 1 to -1.
2.train : dot product of noisy training data is done and generated a weight matrix which is useful later for image retrival
3.retrieve_pattern : Using the weight matrix, one noisy image as input, and also the img size, reconstructs the image (denoising)
4.calculate_similarity : method to find different type of similarities using hamming distance, or cosine similarity based on the type in the function call made
5.select_representative_images : Inorder to train HNN, from each state representative images(which are similar/represents most of images of that state) using KMEANS clustering
6.retrieve_most_similar : noisy image is flattend, and makes function call to reconstruct the image and most similar clean pattern from a set of reference patterns (all_patterns) is identified 
Remaining methods are self explainable

artifact_No_HNN.py
This script is for the (II) task as mentioned above. Noise added to images , without denoising, classification is done
For different combination of noise percent and image sizes, finding classification accuracy and confusion matrix
Methods:
1.insert_artifact : Noise is inserted into images
2.No_denoising : Generate images of readings, add noise, prepare for training and does training and evaluation.

NOTE : Each script runned for only 2 epochs, so less accuracy, but for epochs 10 it gave best results and more good results for further more no. of epochs

