from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
class Support_Vector_Machines_SMO():
    def __init__(self, kernel="linear", C=100, tolerance=1e-5, max_iter=500, gamma=1, degree=2):
        random.seed(17)
        self.max_iter = max_iter
        kernels = {
            'linear': self.kernel_linear,
            'poly': self.kernel_poly,
            'rbf': self.kernel_rbf
        }
        self.kernel_name = kernel
        self.kernel = kernels[self.kernel_name]
        self.C = C
        self.gamma=gamma
        self.degree = degree
        self.tolerance = tolerance
    def fit(self, X, y):
        self.X = X
        self.y = y
        #Obliczamy macierz z funkcją kernelową
        self.K = self.kernel(self.X, self.X)
        self.konwersja_predykcji_na_zera = False
        if(0 in self.y):
            self.y = y * 2 - 1
            self.konwersja_predykcji_na_zera = True
        self.lambdas = np.zeros(self.X.shape[0])
        #Inicjalizacja naszego wyrazu wolnego b
        self.b = 0
        iter = 0
        while True:
            iter = iter + 1
            i = 0
            lambdas_old = self.lambdas.copy()
            while(i < len(self.X)):
                E_i = self.calculate_E_x_i(index=i)
                if((E_i*self.y[i] < -self.tolerance and self.lambdas[i] < self.C) or (self.y[i]*E_i > self.tolerance and self.lambdas[i] > 0)):
                    j = self.get_rnd_int(0, len(self.X)-1, i)
                    #Kopiowanie starych lambd
                    lambda_i_old, lambda_j_old = self.lambdas[i], self.lambdas[j]
                    L, H = self.calculate_L_H(y_i=self.y[i], y_j=self.y[j], C=self.C, lambda_i=self.lambdas[i], lambda_j=self.lambdas[j])
                    #Na wszelki wypadek. Przykładem może być zbiór gdzie punkty obu klas są równomiernie rozłożone na okręgu o dużym promieniu, wówczas może się zdarzyć, że L==H
                    if(L == H):
                        i = i + 1
                        continue
                    eta = self.K[i, i] + self.K[j, j]- 2*self.K[i, j]
                    #Jeżeli eta ==0 to oznacza, że mamy dwie identyczne obserwacje - duplikaty. Jeżeli eta <= 0 to znaczy, że maksymalizujemy funkcje zamiast ją minimalizować
                    if(eta <= 0):
                        i = i + 1
                        continue
                    E_j = self.calculate_E_x_i(index=j)
                    self.lambdas[j] = self.calculate_lambda_j(lambda_j=self.lambdas[j], y_j=self.y[j], E_i=E_i, E_j=E_j, eta=eta)
                    self.lambdas[j] = self.check_bounds(lambda_j=self.lambdas[j], L=L, H=H)
                    #Sprawdzamy czy na pewno nasza nowa lambda mocno się różni od starej, jeżeli nie to zaczynamy od nowego indeksu
                    if(self.lambdas[j] - lambda_j_old < self.tolerance):
                        i = i + 1
                        continue
                    self.lambdas[i] = self.calculate_lambda_i(lambda_i_old=lambda_i_old, lambda_j_old=lambda_j_old, lambda_j_new=self.lambdas[j], y_i=self.y[i], y_j=self.y[j])
                    #Jeżeli znaleźliśmy nowe wartości lambd to możemy zaktualizować wyraz wolny  b
                    self.b = self.calculate_b(i=i, j=j, E_i=E_i, E_j=E_j, lambda_i_new=self.lambdas[i], lambda_i_old=lambda_i_old, lambda_j_new=self.lambdas[j], lambda_j_old=lambda_j_old, C=self.C)
                i = i + 1
                #Jeżeli zbiega do 0 to kończymy
            #Obliczamy odległość euklidesową
            diff = np.linalg.norm(self.lambdas - lambdas_old)
            if(diff < self.tolerance):
                break
            elif(iter > self.max_iter):
                print("Iterations exceeded max_iter")
                break
            iter = iter + 1
        if(self.kernel_name=="linear"):
            self.w = self.calculate_w(lambdas=self.lambdas, x=self.X, y=self.y)
        self.support_vectors_ = self.support_vectors_calculation()
        self.sv = np.squeeze(self.lambdas > 1e-5)
        return self.b
    def support_vectors_calculation(self):
        f_x = np.sum(self.lambdas.reshape(-1,1)*self.y.reshape(-1,1)*self.K, axis=0)+self.b
        indexes = []
        for i in range(len(self.X)):
            if(self.y[i]==1 and f_x[i] - 1 < 1e-2 or self.y[i]==-1 and f_x[i] +1 > -1e-2):
                indexes.append(i)
        return self.X[indexes, :]
    def kernel_linear(self, x_1, x_2):
        return x_1@x_2.T
    def kernel_poly(self, x_1, x_2):
        return (self.gamma*x_1@x_2.T+1)**self.degree
    def kernel_rbf(self, x_1, x_2):
        return np.exp(-self.gamma*(-2*x_1@x_2.T + np.sum(x_1*x_1,1)[:,None] + np.sum(x_2*x_2,1)[None,:]))
    def calculate_L_H(self, y_i, y_j, C, lambda_i, lambda_j):
        if(y_i == y_j):
            L = max(0, -C+lambda_i+lambda_j)
            H = min(C, lambda_i+lambda_j)
        else:
            L = max(0, -lambda_i+lambda_j)
            H = min(C, C-lambda_i+lambda_j)
        return L, H
    def calculate_w(self, lambdas, x, y):
        return x.T@np.multiply(lambdas,y)
    def calculate_b(self, i, j, E_i, E_j, lambda_i_new, lambda_i_old, lambda_j_new, lambda_j_old, C):
        b_one = self.b - E_i+self.y[i]*(lambda_i_new-lambda_i_old)*self.K[i, i]+self.y[j]*(lambda_j_new-lambda_j_old)*self.K[i, j]
        b_two = self.b - E_j+self.y[i]*(lambda_i_new-lambda_i_old)*self.K[i, j]+self.y[j]*(lambda_j_new-lambda_j_old)*self.K[j, j]
        if(lambda_i_new > 0  and lambda_i_new < C):
            return b_one
        elif(lambda_j_new > 0 and lambda_j_new < C):
            return b_two
        else:
            return (b_one+b_two)/2
    def calculate_E_x_i(self, index):
        return np.sum(self.lambdas*self.y * self.K[index, :]) + self.b - self.y[index]
    def calculate_lambda_j(self, lambda_j, y_j, E_i, E_j, eta):
        return lambda_j+(y_j*(E_i-E_j))/eta
    def calculate_lambda_i(self, lambda_i_old, lambda_j_old, lambda_j_new, y_i, y_j):
        return lambda_i_old+y_j/y_i*(lambda_j_old-lambda_j_new)
    def check_bounds(self, lambda_j, L, H):
        if(lambda_j > H):
            return H
        elif(lambda_j < L):
            return L
        else:
            return lambda_j
    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = random.randint(a,b)
            cnt=cnt+1
        return i
    def decision_function(self, X_predict):
        return np.sum(self.y[self.sv].reshape(-1,1)*self.lambdas[self.sv].reshape(-1,1)*self.kernel(self.X[self.sv], X_predict), 0) + self.b
    def predict(self, X_predict):
        if(self.konwersja_predykcji_na_zera == True):
            return np.where(self.decision_function(X_predict) > 0, 1, 0)
        else:
            return np.where(self.decision_function(X_predict) > 0, 1, -1)