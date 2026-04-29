import itertools
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
from PIL import Image


'''
模型定义
'''

class MLP:

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation, learning_rate, weight_decay, decay_rate, step):
        self.W1 = np.random.randn(input_dim, hidden_dim1) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros((1,hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(2.0/hidden_dim1)
        self.b2 = np.zeros((1,hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, output_dim) * np.sqrt(2.0/hidden_dim2)
        self.b3 = np.zeros((1,output_dim))
        self.activation = activation
        self.original_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        self.step = step
        self.lr_history = []
        self.loss_history = []
        self.val_loss_history = []
        self.val_accuracy = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def ReLU(self, z):
        return np.maximum(0, z)

    def ReLU_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def update_learning_rate(self, epoch):
        old_lr = self.current_learning_rate
        new_lr = self.original_learning_rate * self.decay_rate ** (epoch//self.step)
        self.current_learning_rate = new_lr
        self.lr_history.append(old_lr)

    def cross_entropy_loss(self,y_predict, y_true):
        m = y_true.shape[0]
        ll = -np.log(y_predict[np.arange(m), y_true]+1e-8)
        loss = np.sum(ll) / m
        l2_reg = (self.weight_decay / 2) * (np.sum(self.W1 **2) + np.sum(self.W2 **2) + np.sum(self.W3 **2))
        total = loss + l2_reg
        return total

    def forward(self, X):
        if self.activation == 'sigmoid':
            self.z1 = np.dot(X,self.W1)+self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1,self.W2)+self.b2
            self.a2 = self.sigmoid(self.z2)
            self.z3 = np.dot(self.a2,self.W3)+self.b3
            self.output = self.softmax(self.z3)
            return self.output
        elif self.activation == 'relu':
            self.z1 = np.dot(X,self.W1)+self.b1
            self.a1 = self.ReLU(self.z1)
            self.z2 = np.dot(self.a1,self.W2)+self.b2
            self.a2 = self.ReLU(self.z2)
            self.z3 = np.dot(self.a2,self.W3)+self.b3
            self.output = self.softmax(self.z3)
            return self.output
        else:
            return 0

    def backward(self, X, y_true):
        m = X.shape[0]
        y_onehot = np.zeros_like(self.output)
        y_onehot[np.arange(m), y_true]=1
        if self.activation == 'sigmoid':
            dz3 = self.output - y_onehot
            dW3 = (1/m) * np.dot(self.a2.T, dz3)
            db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
            dW3 += self.weight_decay * self.W3
            da2 = np.dot(dz3, self.W3.T)
            dz2 = da2 * self.sigmoid_derivative(self.z2)
            dW2 = (1/m) * np.dot(self.a1.T, dz2)
            db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
            dW2 += self.weight_decay * self.W2
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.sigmoid_derivative(self.z1)
            dW1 = (1/m) * np.dot(X.T, dz1)
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
            dW1 += self.weight_decay * self.W1
            return dW1, db1, dW2, db2, dW3, db3
        elif self.activation == 'relu':
            dz3 = self.output - y_onehot
            dW3 = (1 / m) * np.dot(self.a2.T, dz3)
            db3 = (1 / m) * np.sum(dz3, axis=0, keepdims=True)
            da2 = np.dot(dz3, self.W3.T)
            dz2 = da2 * self.ReLU_derivative(self.z2)
            dW2 = (1 / m) * np.dot(self.a1.T, dz2)
            db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.ReLU_derivative(self.z1)
            dW1 = (1 / m) * np.dot(X.T, dz1)
            db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
            return dW1, db1, dW2, db2, dW3, db3
        else:
            return 0

    def update_params(self, dW1, db1, dW2, db2, dW3, db3):
        self.W1 -= self.current_learning_rate * dW1
        self.W2 -= self.current_learning_rate * dW2
        self.W3 -= self.current_learning_rate * dW3
        self.b1 -= self.current_learning_rate * db1
        self.b2 -= self.current_learning_rate * db2
        self.b3 -= self.current_learning_rate * db3

    def train(self, X, y, epochs, batch_size=32):
        n_samples = X.shape[0]
        n_vals = int(n_samples*0.1)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_vals]
        train_indices = indices[n_vals:]
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        max_accuracy = 0
        model_path = 'model.pkl'
        for epoch in range(epochs):
            n_train = X_train.shape[0]
            indices = np.random.permutation(n_train)
            X_train = X_train[indices]
            y_train = y_train[indices]
            epoch_loss = 0
            n_batches = 0
            for i in range(0, n_train, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                y_pred = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1
                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)
                self.update_params(dW1, db1, dW2, db2, dW3, db3)
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            val_pred = self.forward(X_val)
            val_loss = self.cross_entropy_loss(val_pred, y_val)
            self.val_loss_history.append(val_loss)
            self.update_learning_rate(epoch)
            ac = accuracy_score(y_val, self.predict(X_val))
            self.val_accuracy.append(ac)
            if ac > max_accuracy:
                max_accuracy = ac
                with open(model_path, 'wb') as f:
                    pickle.dump(self, f)
            if (epoch + 1) % 10 == 0:
                print("epoch:%d, training loss: %f, validation loss: %f, validation accuracy: %f"
                    % (epoch+1, avg_loss, val_loss, ac))
        return self.loss_history, self.val_loss_history, self.val_accuracy, model_path


    def predict(self, X):
        y = self.forward(X)
        return np.argmax(y, axis=1)

    def predict_proba(self, X):
        return self.forward(X)

if __name__ == '__main__':
    '''
    数据预处理
    '''
    number_of_category = 10
    sample = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
    category = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    filepath = 'EuroSAT_RGB/'
    X = []
    y = []
    for i in range(number_of_category):
        subdata = []
        fp = filepath + category[i] + '/' + category[i] + '_'
        for j in range(sample[i]):
            image = Image.open(fp + '%d' % (j + 1) + '.jpg')
            image_size = image.size
            img_array = np.array(image, dtype=np.float32)
            img_array /= 255.0
            img_array = img_array.flatten()
            X.append(img_array)
            image.close()
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    print(y.shape[0])
    print(image_size)


    input_dim = X.shape[1]

    '''
    超参数查找
    '''

    params_grid = {
        'hidden_dim': [(64,32),(128,64),(256,128)],
        'learning_rate': [0.005,0.01,0.05],
        'weight_decay': [0,0.0001,0.001]
    }
    param_names = list(params_grid.keys())
    param_values = list(params_grid.values())
    combinations = []
    for values in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, values)))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    for combination in combinations:
        model = MLP(
            input_dim = input_dim,
            hidden_dim1 = combination['hidden_dim'][0],
            hidden_dim2 = combination['hidden_dim'][1],
            output_dim = number_of_category,
            activation = 'relu',
            learning_rate = combination['learning_rate'],
            weight_decay = combination['weight_decay'],
            decay_rate = 0.99,
            step = 5
        )
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], [val_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            tl, vl, va, mp = model.train(X_train, y_train, epochs = 20)
            scores.append(max(va))
        result = {'params':combination, 'score':np.mean(scores)}
        print(result)
        results.append(result)
    results.sort(key=lambda x: x['score'], reverse=True)
    print(results)


    '''
    模型训练与预测
    '''


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLP(
        input_dim = input_dim,
        hidden_dim1 = 256,
        hidden_dim2 = 128,
        output_dim = number_of_category,
        activation = 'relu',
        learning_rate = 0.01,
        weight_decay = 0.001,
        decay_rate = 0.99,
        step = 5
    )

    train_loss, val_loss, val_accuracy, model_path = mlp.train(X_train, y_train, epochs = 50)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)


    y_pred = model.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    print("Accuracy: %.4f"%ac)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(50),train_loss,label='Training Loss')
    plt.plot(range(50),val_loss,label='Validation Loss')
    plt.legend()
    plt.show()
    plt.figure()
    plt.title('Validation Accuracy Curve')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(range(50),val_accuracy,label='Validation Accuracy')
    plt.legend()
    plt.show()



    W1 = model.W1
    for i in range(56,64):
        w = W1[:, i]
        normalized = (w - w.min()) / (w.max() - w.min())
        im_recover = normalized.reshape(image_size[0], image_size[1], 3)
        plt.figure()
        plt.imshow(im_recover)
        plt.axis('off')
        plt.show()
