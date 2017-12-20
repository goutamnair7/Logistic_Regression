import sys
import math
import re

def tokenize(data):
    return re.findall('\\w+', data)

def sigmoid(score):
    overflow = 20.0
    if score > overflow:
        score = overflow
    elif score < -overflow:
        score = -overflow

    exp = math.exp(score)
    return exp / (1+exp)

def main():
    
    vocab_size = int(sys.argv[1])
    eta = float(sys.argv[2])
    mu = float(sys.argv[3])
    max_iter = int(sys.argv[4])
    train_size = int(sys.argv[5])
    test_file = sys.argv[6]

    B = []
    p = []

    A = []

    train_labels = ['Person', 'other', 'Work', 'Species', 'Place']
    for i in range(0, len(train_labels)):
        p.append(0.0)
    
    for i in range(0, vocab_size):
        B.append([])
        A.append([])

    k = 0

    for t in range(1, max_iter+1):
        
        print "iteration: ", t
        lambda_ = eta / (t*t)    
        
        lcl = 0.0

        for q in range(0, train_size):
            k += 1
            train_ex = sys.stdin.readline()
            
            given_labels = train_ex.strip().split("\t")[1].strip()
            data = train_ex.strip().split("\t")[2].strip()
            words = tokenize(data)

            labels = []
            for label in train_labels:
                if label in given_labels:
                    labels.append(1.0)
                else:
                    labels.append(0.0)
            
            X = {}
            for word in words:
                j = hash(word.lower()) % vocab_size
                if X.has_key(j):
                    X[j] += 1
                else:
                    X[j] = 1

            for j in X.keys():
                ct = X[j]
                if len(B[j]) > 0:
                    for i in range(0, len(train_labels)):
                        p[i] += (B[j][i] * ct)
            
            for i in range(0, len(train_labels)):
                p[i] = sigmoid(p[i])

            for i in range(0, len(labels)):
                p[i] += lambda_ * (labels[i] - p[i])
            
            const = 2 * lambda_ * mu
            for j in X.keys():
                
                ct = X[j]
                new_B = [0.0] * 5
                new_A = [0] * 5
                
                if len(B[j]) > 0:
                    new_B = B[j][:]
                    new_A = A[j][:]
                    for i in range(0, len(train_labels)):
                        new_B[i] = new_B[i] * math.pow(1.0 - const, k - new_A[i])

                for i in range(0, len(labels)):
                    new_B[i] += lambda_ * (labels[i] - p[i]) * ct
                    new_A[i] = k
                
                B[j] = new_B[:]
                A[j] = new_A[:]
                
            for i in range(0, len(train_labels)):
                if labels[i] == 1.0:
                    lcl += math.log(p[i])
                else:
                    lcl += math.log(1.0 - p[i])

        const = 2 * lambda_ * mu
        for j in X.keys():
            if len(B[j]) > 0:
                for i in range(0, len(train_labels)):
                    B[j][i] = B[j][i] * math.pow(1.0 - const, k-A[j][i])
                    A[j][i] = k
       
        print "lcl: ", lcl
        avg_lcl = (lcl*1.0)/(5 * train_size)
        print "avg lcl: ", avg_lcl

    sys.exit()    
    f = open(test_file, "r")
    lines = f.readlines()
    f.close()

    for line in lines:
        data = line.strip().split("\t")[2].strip()
        words = tokenize(data)
        
        pred = p[:]
        X = {}
        for word in words:
            j = hash(word.lower()) % vocab_size
            if X.has_key(j):
                X[j] += 1
            else:
                X[j] = 1
        
        for j in X.keys():
            ct = X[j]
            if len(B[j]) > 0:
                for i in range(0, len(train_labels)):
                    pred[i] += (B[j][i] * ct)
        
        for i in range(0, len(train_labels)):
            pred[i] = sigmoid(pred[i])

        sys.stdout.write("Person\t"+str(pred[0])+",")
        sys.stdout.write("other\t"+str(pred[1])+",")
        sys.stdout.write("Work\t"+str(pred[2])+",")
        sys.stdout.write("Species\t"+str(pred[3])+",")
        sys.stdout.write("Place\t"+str(pred[4])+"\n")

if __name__ == "__main__":
    main()
