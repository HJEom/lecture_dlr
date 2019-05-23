import numpy as np

class Logistic_Regression:
    def __init__(self, number_data, learning_rate, iteration):
        self.nd = number_data
        self.it = iteration
        self.lr = learning_rate
    def sigmoid(self, _in):
        return 1/(1+np.exp(-_in))
    def relu(self, _in):
        return _in*(_in>0)
    def logis_reg(self, _input, label, w1, b1, w2, b2):
        z1 = np.dot(_input,w1)+b1   # (128,2)
        act1 = self.sigmoid(z1)
        z2 = np.dot(act1,w2)+b2  # (128,1)
        act2 = self.sigmoid(z2)
        cost = np.sum(-(label)*np.log10(act2)+(label-1)*np.log10(1-act2))/self.nd   # cross entropy
        dz2 = act2 - label   # (128,1)
#        cost = np.sum((label-act2)*(label-act2))/self.nd   # mean squared error
#        dz2 = 2*(act2-label)*act2*(1-act2)   # (128,1)   for mean squared error
        dw2 = (np.dot(act1.T,dz2))/self.nd   # (2,1)
        db2 = (np.sum(dz2, axis=0, keepdims=True))/self.nd
        dz1 = (np.dot(dz2,w2.T))*(self.sigmoid(z1)*(1-self.sigmoid(z1)))   # (128,2)
        dw1 = (np.dot(_input.T,dz1))/self.nd   # (2,2)
        db1 = (np.sum(dz1, axis=0, keepdims=True))/self.nd
        w2 = w2 - self.lr*dw2
        b2 = b2 - self.lr*db2
        w1 = w1 - self.lr*dw1
        b1 = b1 - self.lr*db1
        return w1, b1, w2, b2, cost
    def accuracy(self, _input, label, w1, b1, w2, b2):
        correct = 0
        act = self.sigmoid(np.dot(self.sigmoid(np.dot(_input,w1)+b1),w2)+b2)
        for i in range(self.nd*self.nd):
            if ((act[i][0] >= 0.5) & (label[i][0]==1)):
                correct += 1
            elif ((act[i][0] < 0.5) & (label[i][0]==0)):
                correct += 1
        return float(correct)/(float(self.nd)*float(self.nd))

# hyper parameter
number_data = 128
lr = 0.1
iteration = 5000
# container
training_set = np.zeros((number_data, 2), dtype='f')
training_label = np.zeros((number_data,1),dtype='f')
test_set = np.zeros((number_data*number_data, 2), dtype='f')
test_label = np.zeros((number_data*number_data,1),dtype='f')

# for step 6 & step 7
mean_prediction_acc = 0.0
best_acc = 0.0
best_w1 = np.zeros((2,2), dtype='f')
best_b1 = np.zeros((2), dtype='f')
best_w2 = np.zeros((2,1),dtype='f')
best_b2 = np.zeros((1),dtype='f')

# construct net
net = Logistic_Regression(number_data, lr, iteration)

####### step 6. repeat step 1~6 for 10 times
for mean_acc in range(10):
    # container
    w1 = np.zeros((2,2), dtype='f')
    b1 = np.zeros((2), dtype='f')
    w2 = np.zeros((2,1),dtype='f')
    b2 = np.zeros((1), dtype='f')
    # initialize weights & bias
    for cur_n1 in range(2):
        for pre_n1 in range(2):
            w1[pre_n1][cur_n1] = np.random.uniform(-1.0, 1.0)
        b1[cur_n1] = np.random.uniform(-1.0, 1.0)
    for cur_n2 in range(1):
        for pre_n2 in range(2):
            w2[pre_n2][cur_n2] = np.random.uniform(-1.0, 1.0)
        b2[cur_n2] = np.random.uniform(-1.0,1.0)

    ####### step 4. repeat step 1~3 for 5000 times
    for itr in range(iteration+1):
        
        ####### step 1. generate data set for training
        for nd in range(number_data):
            training_set[nd][0] = np.random.uniform(-1.0,1.0)
            training_set[nd][1] = np.random.uniform(-1.0,1.0)
        '''
        # data pre-processing
        avr0 = 0
        avr1 = 0
        for i in range(number_data):
            avr0 += training_set[i][0]
            avr1 += training_set[i][1]
        avr0 = avr0/number_data
        avr1 = avr1/number_data
        var0 = 0
        var1 = 0
        for i in range(number_data):
            var0 += (training_set[i][0]-avr0)*(training_set[i][0]-avr0)
            var1 += (training_set[i][1]-avr1)*(training_set[i][1]-avr1)
        var0 = np.sqrt(var0/number_data)
        var1 = np.sqrt(var1/number_data)
        for i in range(number_data):
            training_set[i][0] = (training_set[i][0]-avr0)/var0
            training_set[i][1] = (training_set[i][1]-avr1)/var1
        '''
        ####### step 2. assign label for training
        for nd in range(number_data):
            if training_set[nd][1] > training_set[nd][0]*training_set[nd][0] :
                training_label[nd][0] = 0
            else:
                training_label[nd][0] = 1
        
        ####### step 3. train
        w1, b1, w2, b2, cost = net.logis_reg(training_set, training_label, w1, b1, w2, b2)

        if itr%1000 == 0:
            print(itr, 'cost : ', str(cost)[:7])
        
    ####### step 1. generate data set for test
    for nd in range(number_data*number_data):
        test_set[nd][0] = np.random.uniform(-1.0,1.0)
        test_set[nd][1] = np.random.uniform(-1.0,1.0)
    '''
    # data pre-processing
    avr0 = 0
    avr1 = 0
    for i in range(number_data*number_data):
        avr0 += test_set[i][0]
        avr1 += test_set[i][1]
    avr0 = avr0/(number_data*number_data)
    avr1 = avr1/(number_data*number_data)
    var0 = 0
    var1 = 0
    for i in range(number_data*number_data):
        var0 += (test_set[i][0]-avr0)*(test_set[i][0]-avr0)
        var1 += (test_set[i][1]-avr1)*(test_set[i][1]-avr1)
    var0 = np.sqrt(var0/(number_data*number_data))
    var1 = np.sqrt(var1/(number_data*number_data))
    for i in range(number_data*number_data):
        test_set[i][0] = (test_set[i][0]-avr0)/var0
        test_set[i][1] = (test_set[i][1]-avr1)/var1
    '''
    ####### step 2. assign label for test
    for nd in range(number_data*number_data):
        if test_set[nd][1] > test_set[nd][0]*test_set[nd][0] :
            test_label[nd][0] = 0
        else:
            test_label[nd][0] = 1
    
    ####### step 5. calculate accuracy
    acc = net.accuracy(test_set, test_label, w1, b1, w2, b2)
    print(mean_acc, 'accuracy : ', acc)

    ####### step 6. calculate the mean prediction accuracies
    mean_prediction_acc += acc
    
    ####### step 7. the best of trained weights & bias
    if acc > best_acc:
        best_acc = acc
        best_w1 = w1
        best_b1 = b1
        best_w2 = w2
        best_b2 = b2
print('\n version : Numpy')
print('\n mean_accuracy : ', str(mean_prediction_acc/10)[:5])
print('\n best accuracy : ', best_acc)
print('\n best w1')
print(best_w1)
print('\n best b1')
print(best_b1)
print('\n best w2')
print(best_w2)
print('\n best b2')
print(best_b2)
