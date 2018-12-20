'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for each_class in range(0,10):
        class_digits = data.get_digits_by_label(train_data,train_labels,each_class)
        class_mean = (np.sum(class_digits,axis=0))/(class_digits.shape[0])
        means[each_class] += class_mean
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data,train_labels)

    for each_class in range(0,10):
        # According to the covariance formula, we calculated covariance as following.
        class_digits = data.get_digits_by_label(train_data, train_labels, each_class)
        difference = class_digits - means[each_class]
        each_class_cov = (np.dot(difference.T,difference))/len(class_digits)
        covariances[each_class,:,:] = each_class_cov + 0.01 *np.identity(class_digits.shape[1])
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''

    log_gen_likelihood = np.zeros((digits.shape[0],10))
    for each_class in range(0, 10):
        # Because this is an log generative likelihood, after simplify
        # we have the following terms.
        first_term = np.log((2*np.pi)**(-digits.shape[1]/2))

        second_term = np.log(np.linalg.det(covariances[each_class])**(-1/2))

        difference = digits - means[each_class]
        inv_cov = np.linalg.inv(covariances[each_class])

        third_term = np.log(np.exp(np.diag((-1/2)*difference.dot(inv_cov).dot(difference.T))))

        log_gen_likelihood[:,each_class] = first_term + second_term + third_term

    return log_gen_likelihood


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    log_p_x_y = generative_likelihood(digits,means,covariances)
    log_p_y = np.log(1/10)
    log_p_x = np.log(np.sum(np.exp(log_p_x_y)*0.1,axis=1)).reshape(-1,1)
    log_p_y_x = log_p_x_y + log_p_y - log_p_x
    return log_p_y_x



def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    total = 0
    for i in range (0,digits.shape[0]):
        total += cond_likelihood[i][int(labels[i])]
    avg_cond_likelihood = (total)/(digits.shape[0])
    return avg_cond_likelihood


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    # Pick the max as most likely posterior class.
    most_likely_class = np.argmax(cond_likelihood,axis=1)

    return most_likely_class




def accuracy(digits, labels, means, covariances):
    '''
    Return the accuracy according to most likely posterior
    class for each training and test data point as
    prediction.
    '''
    most_likely_class = classify_data(digits,means,covariances)
    correct_classify = 0
    for i in range(0,most_likely_class.shape[0]):
        # If the class we chosen is correct.
        if (most_likely_class[i] == labels[i]):
            correct_classify += 1
    res_accuracy = correct_classify / most_likely_class.shape[0]
    return res_accuracy



def plot(covariances):
    from numpy import linalg as LA
    res = []
    for i in range(0,10):
        eigenvalues,eigenvectors = LA.eig(covariances[i])
        temp = eigenvectors[:,0].reshape((8,8))
        res.append(temp)
    concat = np.concatenate(res, 1)
    plt.imshow(concat, cmap='gray')
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    # print(means)
    covariances = compute_sigma_mles(train_data, train_labels)
    #print(covariances)

    log__generative_likelihood = generative_likelihood(train_data,means,covariances)
    # print(log__generative_likelihood)

    log_conditional_likelihood = conditional_likelihood(train_data,means,covariances)
    # print(log_conditional_likelihood)

    # Evaluation
    # plot(covariances)
    avg_cond_train = avg_conditional_likelihood(train_data,train_labels,means,covariances)
    print("Average conditional log-likelihood for train set is: "+ str(avg_cond_train))

    avg_cond_test = avg_conditional_likelihood(test_data,test_labels,means,covariances)
    print("Average conditional log-likelihood for test set is: " + str(avg_cond_test))

    train_accur = accuracy(train_data,train_labels,means,covariances)
    print("train set accuracy is: "+ str(train_accur))

    test_accur = accuracy(test_data,test_labels,means,covariances)
    print("test set accuracy is: "+ str(test_accur))



if __name__ == '__main__':
    main()
