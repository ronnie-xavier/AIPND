#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# DONE: 0. Fill in your information in the programming header below
# PROGRAMMER:Ronnie Chacko
# DATE CREATED: 16 Dec 2018
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: Previous 05/14/2018 - added import statement that imports the print 
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()
    
    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
#    print(in_arg)
    
    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)
#    print(answers_dic)
#    --example is {'cat_01.jpg': 'cat', 'Poodle_07927.jpg': 'poodle',

    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch )
#    print(result_dic)
#    --example is {'Poodle_07927.jpg': ['poodle', 'standard poodle', 1]}

    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile )
#    print(result_dic)

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)
#    print (results_stats_dic)

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", tot_time)

    # Prints overall runtime in format hh:mm:ss
    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" +
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" + 
          str( int(  ( (tot_time % 3600) % 60 ) ) ) )


# TODO: 2.-to-7. Define all the function below. Notice that the input 
# parameters and return values have been left in the function's docstrings. 
# This is to provide guidance for achieving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to achieve the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: that's a path to a folder
    parser.add_argument('--dir', type = str, default = 'pet_images/', 
                    help = 'Path to the pet image files')
# help text will appear if the user types program name -h

    # Argument 2: that's the CNN model architecture
    parser.add_argument('--arch', type = str, default = 'resnet', 
                    help = 'CNN model architecture to use for image classification')
    
    # Argument 3: that's a text file containing dog labels
    parser.add_argument('--dogfile', type = str, default = 'dognames.txt', 
                    help = 'Text file that contains all labels associated to dogs')

    # Argument that's an integer
    # parser.add_argument('--num', type = int, default = 1,
                  # help = 'Number (integer) input') 
    
    # Assigns variable in_args to parse_args()
    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these labels as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    
    # Retrieve the filenames from folder pet_images/
    filename_list = listdir(image_dir)

    # Print 10 of the filenames from folder pet_images/
#    print("\nPrints 10 filenames from folder pet_images/")
#    for idx in range(0, 10, 1):
#        print("{} file: {}".format (idx + 1, filename_list[idx] ))

    petname_list = []
    for idx in range (0,len(filename_list)):
        low_pet_image = filename_list[idx].lower()

        # Splits lower case string by _ to break into words 
        word_list_pet_image = low_pet_image.split("_")

        # Create pet_name starting as empty string
        pet_name = ""

        # Loops to check if word in pet name is only
        # alphabetic characters - if true append word
        # to pet_name separated by trailing space 
        for word in word_list_pet_image:
            if word.isalpha():
                pet_name += word + " "

        # Strip off starting/trailing whitespace characters 
        pet_name = pet_name.strip()

        # Prints resulting pet_name
#        print("\nFilename=", filename_list[idx], "   Label=", pet_name)
        
        petname_list.append(pet_name)
        
#    print(petname_list)

                      
    # Creates empty dictionary named pet_dic
    pet_dic = dict()

    # Determines number of items in dictionary
#    items_in_dic = len(pet_dic)
#    print("\nEmpty Dictionary pet_dic - n items=", items_in_dic)

    # Adds new key-value pairs to dictionary ONLY when key doesn't already exist
    keys = filename_list
    values = petname_list
    for idx in range(0, len(keys), 1):
        if keys[idx] not in pet_dic:
             pet_dic[keys[idx]] = values[idx]
        else:
             print("** Warning: Key=", keys[idx], 
                   "already exists in pet_dic with value =", pet_dic[keys[idx]])

    #Iterating through a dictionary printing all keys & their associated values
#    print("\nPrinting all key-value pairs in dictionary pet_dic:")
#    for key in pet_dic:
#        print("Key=", key, "   Value=", pet_dic[key])

    return pet_dic


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its key is the
                     pet image filename & its value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()

    # Populates empty dictionary with both labels &indicates if they match (idx 2)
    for pet_img_file in petlabel_dic:
        # If first time key is assigned initialize the list with pet & 
        # classifier labels
        if pet_img_file not in results_dic:
            image_classification = classifier(images_dir+pet_img_file, model)
            image_classification = image_classification.lower().strip() # removes leading or trailing spaces, if any
            results_dic[pet_img_file] = [ petlabel_dic [pet_img_file], image_classification ]

            # Determine if pet_labels matches classifier_labels using find() string 
            # function - so if pet label is found within classifier label it's a match
            
            found = image_classification.find(petlabel_dic[pet_img_file])

            # Key already exists because labels were added, append value to end of
            # list for idx 2 
            # if pet image label was FOUND then there is a match 
            if found >= 0:
                results_dic[pet_img_file] += [ 1 ]

            # if pet image label was NOT found then there is no match
            else:
                results_dic[pet_img_file] += [ 0 ]

    return results_dic


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line.
                Dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    dogsfile_dict = dict()
    
    with open(dogsfile,'r') as f:
        for line in f:
            words = line.split(',')
            for word in words:
                word=word.strip()
                if word not in dogsfile_dict:
                    dogsfile_dict[word] = 1

#    print(dogsfile_dict)

#    print(results_dic)
    for pet_img_file in results_dic:
        if results_dic[pet_img_file][0]  in dogsfile_dict:
            results_dic[pet_img_file] += [1]
        else:
            results_dic[pet_img_file] += [0]

        classifier_is_a_dog = False
        for dog_name in results_dic[pet_img_file][1].split(','):
            dog_name = dog_name.strip()
            if dog_name in dogsfile_dict:
                classifier_is_a_dog = True
            
        if classifier_is_a_dog:
            results_dic[pet_img_file] += [1]
        else:
            results_dic[pet_img_file] += [0]

#    print(results_dic)

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    
    results_stats = dict()
    n_dog_images, n_nondog_images, n_correct_dogs = 0,0,0
    n_correct_nondogs, n_correct_breed, n_correct_label  = 0,0,0

    n_images = len(results_dic)
    
    
    for key in results_dic:
        if results_dic[key][3] == 1:
            n_dog_images += 1
        if results_dic[key][3] == 1 and results_dic[key][4] == 1:
            n_correct_dogs += 1
        if results_dic[key][3] == 0 and results_dic[key][4] == 0:
            n_correct_nondogs += 1
        if results_dic[key][2]==1 and results_dic[key][3]==1:
            n_correct_breed += 1
            
        if results_dic[key][2]==1:            
            n_correct_label += 1
        
        
        
    n_nondog_images = n_images - n_dog_images
    
    results_stats ['n_images'] = n_images
    results_stats ['n_dog_images'] = n_dog_images
    results_stats ['n_nondog_images'] = n_nondog_images
    
    results_stats ['n_correct_dogs'] = n_correct_dogs
    results_stats ['n_correct_nondogs'] = n_correct_nondogs

    results_stats ['n_correct_breed'] = n_correct_breed
    results_stats ['n_correct_label'] = n_correct_label

    percentage_calculator = lambda x,y: x*100/y if y != 0 else 0

    results_stats ['pct_correct_dogs'] = percentage_calculator (n_correct_dogs, n_dog_images)
    results_stats ['pct_correct_nondogs'] = percentage_calculator (n_correct_nondogs, n_nondog_images)
    results_stats ['pct_correct_breed'] = percentage_calculator (n_correct_breed, n_dog_images)
    results_stats ['pct_correct_label'] = percentage_calculator (n_correct_label, n_images)
    
    return results_stats


def print_results(results_dic, results_stats, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """
    
    print('     The model {} correctly classified {}% of the images of dogs as dogs! It also correctly classified the breeds in {}% of the cases. Overall, it was able to label {}% of the entire population. The total images were {} out of which {} were dogs'.format(model, results_stats['pct_correct_dogs'], results_stats['pct_correct_breed'],results_stats['pct_correct_label'],
                     results_stats['n_images'],results_stats['n_dog_images']))

    if print_incorrect_dogs:
        print ("dogs not classified as dogs are: ")
        for key in results_dic:
            if results_dic[key][3]==1 and results_dic[key][4]==0:
                print (key)
                print (results_dic[key])
    
    if print_incorrect_breed:
        print ("misclassified breeds are: ")
        for key in results_dic:
            if results_dic[key][3]==1 and results_dic[key][2]==0:
                print (key)
                print (results_dic[key])

                
# Call to main function to run the program

if __name__ == "__main__":
    main()
