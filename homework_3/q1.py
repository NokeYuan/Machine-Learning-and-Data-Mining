import matplotlib.pyplot as plt
import numpy as np

def huber_function(a,delta):
    if abs(a) < delta or abs(a) == delta:
        return 0.5 * (a ** 2)
    if abs(a) > delta:
        return delta * (abs(a)- 0.5 * delta)

def huber_loss(y,t,delta):
    res = y - t
    this = huber_function(res,delta)
    return this

def square_error_loss(y,t):
    res = y - t
    return 0.5 * (res ** 2)


def gradient_descent(x,y,w,b,delta):

    w_matrix = w
    b_value = b
    iteration = 0

    # Make a threshold that iteration can only run for 10000 times.
    while (iteration < 10000):

        #Using w and b, do computation on design matrix X
        result = np.dot(x,w_matrix)+b_value
        #Compare result with target Y
        difference = result - y
        N = difference.shape[0]


        d_t = difference.T
        #according to the result from Q1 (b), we have 3 differen cases to set dj/dw.
        each_dj_dw = np.piecewise(d_t,[abs(d_t)<= delta, d_t >delta, d_t < -delta ],[lambda d_t: d_t, lambda d_t: delta, lambda d_t: -delta])
        each_dj_dw_with_x = np.dot(each_dj_dw, x)
        #compute the average w
        dj_dw = each_dj_dw_with_x / N

        # according to the result from Q1 (b), we have 3 differen cases to set dj/db.
        each_dj_db = np.piecewise(d_t, [abs(d_t) <= delta, d_t > delta, d_t < -delta],[lambda d_t: d_t, lambda d_t: delta, lambda d_t: -delta])
        # compute the average w
        dj_db = np.sum(each_dj_db) / N

        # if np.all(dj_dw, [0]) and np.all(dj_db, [0]):
        if np.sum(dj_dw) + dj_db == 0:
            break

        #update w matrix and b value
        w_matrix = w_matrix - 0.001 * dj_dw
        b_value = b_value - 0.001 * dj_db

        iteration += 1
    return w_matrix, b_value




if __name__ == "__main__":

# graph for Q1 (a)

    t = np.arange(-50,50,0.5)
    new_y = ((t * 3) + 1).reshape(-1, 1)
    loss = []
    for item in t:
        temp = huber_loss(item,0,5)
        loss.append(huber_loss(item,0,5))
    print(loss)
    plt.plot(t,loss)


    square_loss = []
    for item in t:
        square_loss.append(square_error_loss(item,0))

    plt.plot(t,square_loss)
    plt.show()



#-----------------testing for gradient descent----------------------
    # x = t.reshape(-1,1)
    # y = np.array(loss).reshape(-1, 1)
    # w_matrix = np.zeros((1,1))
    # b = 0
    # delta = 10
    # w, b = gradient_descent(x, new_y, w_matrix, b, delta)
    # print(x)
    # print(new_y)
    # print(w_matrix)
    # print(w)
    # print(b)




