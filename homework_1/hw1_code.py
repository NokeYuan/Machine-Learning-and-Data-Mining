from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import random
import math
import numpy
import graphviz


def load_data():

    #open file for reading
    with open ("clean_real.txt","r") as real:
        real_content = real.readlines()
    real_content = [x.strip() for x in real_content]

    with open ("clean_fake.txt","r") as real:
        fake_content = real.readlines()
    fake_content = [x.strip() for x in fake_content]

    # Make dictionary to stores each item in real_content and fake_content as key,
    # and lable 'TRUE' or 'FALSE' as its value.
    my_dict = {}
    for item in real_content:
        my_dict[item] = "TRUE"

    for item in fake_content:
        my_dict[item] = "FALSE"

    #Shuffle , make it random.
    keys = list(my_dict.keys())
    random.shuffle(keys)

    #Vectorizer, I use countvectorizer() here.
    vectorizer = CountVectorizer()
    corpus = vectorizer.fit_transform(keys)
    result = corpus.toarray()

    # Find the length of result array.
    all_index = len(result)
    # Find the index of item ,which at 70% of result array.
    seventy = math.floor(all_index * 0.70)
    # Find the index of item ,which at 85% of result array.
    eighty_five = math.floor(all_index * 0.85)

    # Get featured names.
    featured_name = vectorizer.get_feature_names()

    # Splits the entire dataset into 70% training, 15% vali- dation, and 15% test examples.
    # The dataset is random , becasue it was shuffled.
    training_set = result[:seventy]
    validation_set = result[seventy:eighty_five]
    test_set = result[eighty_five:]

    # Make a list of lables, it follows the order of data after shuffle.
    lables = []
    for item in keys:
        lables.append(my_dict[item])

    # Split the lable.
    training_set_lable = lables[:seventy]
    validation_set_lable = lables[seventy:eighty_five]
    test_set_lable = lables[eighty_five:]

    #Return everything as a list.
    return [training_set,training_set_lable,validation_set,validation_set_lable,test_set,test_set_lable,featured_name]


def select_model(all_data_set, max_depth):

    training_set = all_data_set[0]
    training_set_lable = all_data_set[1]
    validation_set = all_data_set[2]
    validation_set_lable = all_data_set[3]
    total = len(validation_set)

    # For each max_depth, build a DecisionTreeClassifier in gini.
    for number in max_depth:
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth = number)
        clf.fit(training_set, training_set_lable)
        correct = 0
        i = 0
        for item in validation_set:
            if clf.predict([item])[0] == validation_set_lable[i]:
                correct += 1
            i += 1
        accuracy = correct/total
        print("criterion='gini', max_depth =" + str(number) + ",correct = " + str(correct) + ",total=" + str(total) +",accuracy =" + str(accuracy))

    # For each max_depth, build a DecisionTreeClassifier in entropy.
    for number in max_depth:
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth = number)
        clf.fit(training_set, training_set_lable)
        correct = 0
        i = 0
        for item in validation_set:
            if clf.predict([item])[0] == validation_set_lable[i]:
                correct += 1
            i += 1
        accuracy = correct/total
        print("criterion='entropy', max_depth =" + str(number) + ",correct = " + str(correct) + ",total=" + str(total) +",accuracy =" + str(accuracy))



def compute_information_gain(given_featured_name,featured_name,training_set,training_set_lable):

    #Find index of given featured name.
    root_index = featured_name.index(given_featured_name)

    #Get information of true and false in total training set.
    total_training = len(training_set)
    total__training_true = 0
    total__training_false = 0

    for item in training_set_lable:
        if item == "TRUE":
            total__training_true += 1
        elif item == "FALSE":
            total__training_false += 1

    #Get information of true and false when root's occurrence is greater than 0 (right node).
    index_right = 0
    count_total_right = 0
    count_true_right = 0
    count_false_right = 0
    for item in training_set:
        if item[root_index] != 0:
            count_total_right += 1
            if training_set_lable[index_right] == "TRUE":
                count_true_right += 1
            elif training_set_lable[index_right] == "FALSE":
                count_false_right += 1
        index_right += 1

    #Get information of true and false when root's occurrence is less than 0 (left node).
    index_left = 0
    count_total_left = 0
    count_true_left = 0
    count_false_left = 0
    for item in training_set:
        if item[root_index] == 0:
            count_total_left += 1
            if training_set_lable[index_left] == "TRUE":
                count_true_left += 1
            elif training_set_lable[index_left] == "FALSE":
                count_false_left += 1
        index_left += 1

    # Find entropy for root.
    entropy_root = -(total__training_false / total_training) * math.log(total__training_false / total_training, 2) - \
                   (total__training_true / total_training) * math.log(total__training_true / total_training, 2)

    # Find entropy for right node.
    entropy_right = -(count_false_right/count_total_right)*math.log(count_false_right/count_total_right,2) - \
                    (count_true_right/count_total_right)*math.log(count_true_right/count_total_right,2)

    # Find entropy for left node.
    entropy_left = -(count_false_left/count_total_left)*math.log(count_false_left/count_total_left,2) - \
                   (count_true_left/count_total_left)*math.log(count_true_left/count_total_left,2)

    # Calculate information gain for topmost.
    information_gain = entropy_root - (count_total_right/total_training)*entropy_right - \
                       (count_total_left/total_training)*entropy_left

    return information_gain


if __name__ == "__main__":



    # ------------------- Question 2 (a) ---------------
    #Load all data into all_data_set.
    all_data_set = load_data()

    print("--------------- Question 2 (b) --------------- \n")
    max_depth = [1,3,5,7,9]
    select_model(all_data_set,max_depth)
    print("\n")

    print("--------------- Question 2 (d) --------------- \n")

    #Get training set.
    training_set = all_data_set[0]
    #Get training set lable
    training_set_lable = all_data_set[1]
    #Get all featured name
    featured_name = all_data_set[6]
    #Pick given featured name. In this question, I choose 'donal' as root featured name.
    given_featured_name = 'donald'
    print("Information gain for "+ given_featured_name + ":" +
          str(compute_information_gain(given_featured_name,featured_name,training_set,training_set_lable)))

    # ------------------- Question 2 (c) ---------------
    # Generate visualization of decision tree.
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=9)
    clf.fit(training_set, training_set_lable)

    dot_data = tree.export_graphviz(clf,out_file="out_one",filled=True, max_depth=2,
                                    class_names= numpy.array(["TRUE", "FALSE"]),
                                    feature_names =all_data_set[6])
    graph = graphviz.Source(dot_data)
    graph

