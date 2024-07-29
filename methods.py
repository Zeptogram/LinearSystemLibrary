"""
    FILE METHODS.PY 

    AUTORI: 
    - Biancini Mattia 865966
    - Gargiulo Elio 869184

    DESCRIZIONE:
    File/Libreria che contiene le vere e proprie implementazioni dei quattro metodi risolutivi,
    con debugging e funzioni ausiliari all'implementazione complessiva. 

"""
# Per directories
import os
import shutil
# Per tempo, numeri casuali 
from datetime import datetime, timedelta
from random import randrange
from typing import Any
# Per operazioni vettori, matrici
import numpy as np
import scipy.sparse as ss
# Per debugging soluzione
import scipy.sparse.linalg as ssl

""" 
    CLASSE: LinearSystemSolver

    DESCRIZIONE:
    La classe che contiene l'implementazione e quindi il cuore di tutta la libreria

    - Costruttore: __init__
    - Metodi Principali:
        - jacobi(self) -> tuple[Any, timedelta, float, int]
        - gauss_seidel(self) -> tuple[Any, timedelta, float, int]
        - gradient(self) -> tuple[Any, timedelta, float, int]
        - conjugate_gradient(self) -> tuple[Any, timedelta, float, int]

"""

class LinearSystemSolver:
    # Lista di tolleranze utilizzate
    tol = [1e-4, 1e-6, 1e-8, 1e-10]

    """ 
    FUNZIONE:  __init__(self, matrix, iteration, debug, tol_index=None, tol=0):

    DESCRIZIONE: 
    Costruttore della classe LinearSystemSolver che necessita dei seguenti parametri:
        - matrix: la matrice su cui applicare un metodo
        - iteration: numero di iterazioni massimo per un metodo
        - debug: flag per abilitare il debug
        - tol_index: indice per selezionare una tolleranza dalla lista, di default nessuno
        - tol: valore specifico di tolleranza, di default = 0

    """

    def __init__(self, matrix, iteration, debug, tol_index=None, tol=0):
        # Assegnazione dei parametri
        self.prefix = '[DEBUG] '
        self.matrix = ss.csr_matrix(matrix)
        self.iteration = iteration
        self.debug = debug
        # Se non è stato specificato un indice, ma un valore di tolleranza CORRETTO
        if tol_index is None and tol != 0:
            self.tol = tol  # Assegno il valore di tolleranza
        # Se non è stato specificato un indice, ma un valore di tolleranza ERRATO
        elif tol_index is None or tol_index < 0 or tol_index > len(self.tol):
            self.tol = self.tol[randrange(len(self.tol))]  # Scelgo casualmente dalla lista
        # Altrimenti ho un indice specifico con cui selezionare la tolleranza dalla lista
        else:
            self.tol = self.tol[tol_index]
        # Ottengo righe e colonne della matrice
        self.row, self.column = self.matrix.shape
        # Calcolo b = Ax, ovvero il prodotto scalare tra la matrice e x = [1,...1] soluzione esatta
        self.b = self.matrix.dot(_get_solution(self.column))
        # Setup per print che mostra tutta la soluzione
        np.set_printoptions(threshold=np.inf)
        # Setup di flags nel caso sia abilitato o meno il debug
        if debug:
            self.debug_jacobi = True
            self.debug_gauss_seidel = True
            self.debug_gradient = True
            self.debug_conjugate_gradient = True
        else:
            self.debug_jacobi = False
            self.debug_gauss_seidel = False
            self.debug_gradient = False
            self.debug_conjugate_gradient = False

        self.createLog()

    """ 
    FUNZIONE:  jacobi(self) -> tuple[Any, timedelta, float, int]:

    DESCRIZIONE: 
    Funzione che implementa il Metodo di Jacobi: x(k+1) = x(k) + P^(-1) ⋅ r(k)
    Metodo iterativo (risoluzione del problema del fill-in) stazionario che segue una 
    strategia di splitting, ovvero basato sulla decomposizione della matrice A = P - N.

    L'idea alla base del metodo di Jacobi è quella di risolvere il sistema riscrivendo ciascuna 
    equazione in modo che ogni variabile xi sia espressa in funzione delle altre variabili e 
    dei valori della matrice e del vettore dati.

    Il metodo di Jacobi converge sicuramente se la matrice utilizzata è a dominanza diagonale stretta per righe

    La matrice è assunta simmetrica e definita positiva

    """
    def jacobi(self) -> tuple[Any, timedelta, float, int]:
        # Per debugging
        if self.debug_jacobi:
            self.prefix = '[DEBUG-JACOBI] '
        # Definisco il vettore iniziale nullo come inizio del metodo iterativo
        self.x = _get_start_vector(self.column)
        # Ottengo il tempo iniziale di esecuzione
        self._start_time()
        # Ottengo P^(-1)
        reverse_P =  self._getReverseP_jacobi()
        # Controllo se la matrice è a dominanza diagonale stretta per righe
        if not self._row_diagonal_dominance():
            print('[ALERT JB] La matrice fornita non è a dominanza diagonale stretta per righe -> Convergenza non assicurata.')
        # Ciclo for parte da i=0 fino a iteration-1
        for i in range(self.iteration):
            # Calcolo il residuo come r(k) = b − Ax(k)
            residual = self.b - self.matrix.dot(self.x)
            # Calcolo il passo successivo: x(k+1) = x(k) + P^(-1) ⋅ r(k)
            self.x = self.x + reverse_P.dot(residual)
            # Se il debug è attivo stampo l'interazione corrente
            if self.debug_jacobi:
                self._writeIteration(i)
            # Se arrivo alla tolleranza posso fermare l'iterazione
            if self._isTolerate(self.x, self.b):
                if self.debug_jacobi:
                    self._writeTolerance()
                # Ottengo il tempo finale di esecuzione
                self._end_time()
                i += 1  # Incremento per contare l'ultima iterazione
                return self.x, self._total_time(), self.relative_error(self.x), i
        # Se arrivo a max iterazioni
        if self.debug and not self.debug_jacobi:
            self._writeSolution()
        # Ottengo il tempo finale di esecuzione
        self._end_time()
        return self.x, self._total_time(), self.relative_error(self.x), self.iteration

    """ 
    FUNZIONE:  def gauss_seidel(self) -> tuple[Any, timedelta, float, int]:

    DESCRIZIONE: 
    Funzione che implementa il Metodo di Gauss Seidel: x(k+1) = x(k) + P^(-1) ⋅ r(k)

    Il metodo di Gauss Seidel è una variante del metodo di Jacobi, dove vengono sfruttate
    le entrate del vettore x già calcolate, applicando per il calcolo di P^-1 la
    sostituzione in avanti.

    Come Jacobi, il metodo di Gauss Seidel converge sicuramente se la matrice utilizzata è a 
    dominanza diagonale stretta per righe

    La matrice è assunta simmetrica e definita positiva

    """
    def gauss_seidel(self) -> tuple[Any, timedelta, float, int]:
        # Per debugging
        if self.debug_gauss_seidel:
            self.prefix = '[DEBUG-GS] '
        # Definisco il vettore iniziale nullo come inizio del metodo iterativo
        self.x = _get_start_vector(self.column)
        # Ottengo il tempo iniziale di esecuzione
        self._start_time()
        # Calcolo la matrice triangolare inferiore P
        P = self._getP_gauss()
        if not self._row_diagonal_dominance():
            print('[ALERT GS] La matrice fornita non è a dominanza diagonale stretta per righe -> Convergenza non assicurata.')
        # Ciclo for parte da i=0 fino a iteration-1
        for i in range(self.iteration):
            # Se il debug è attivo
            if self.debug_gauss_seidel:
                self._writeIteration(i)
            # Calcolo il residuo come r(k) = b − Ax(k)
            residual = self.b - self.matrix.dot(self.x)
            # Ricavo y applicando la sostituzione in avanti
            y = self._forward_substitution(P, residual)
            # Aggiorno il valore della soluzione x(k+1) = x(k) + y
            self.x = self.x + y
            # Se arrivo alla tolleranza
            if self._isTolerate(self.x, self.b):
                if self.debug_gauss_seidel:
                    self._writeTolerance()
                # Ottengo il tempo finale di esecuzione
                self._end_time()
                i += 1  # Incremento per contare l'ultima iterazione
                return self.x, self._total_time(), self.relative_error(self.x), i
        # Se arrivo a max iterazioni
        if self.debug and not self.debug_gauss_seidel:
            self._writeSolution()
        # Ottengo il tempo finale di esecuzione
        self._end_time()
        return self.x, self._total_time(), self.relative_error(self.x), self.iteration

    """ 
    FUNZIONE:  gradient(self) -> tuple[Any, timedelta, float, int]:

    DESCRIZIONE: 
    Funzione che implementa il Metodo del Gradiente: x(k+1) = x(k) + alpha(k) ⋅ r(k)

    Il metodo del Gradiente è un metodo iterativo non stazionario, dove lo scalare alpha
    dipenderà dalle iterazioni, invece di restare fisso. Esso si basa sulla ricerca di un
    punto di minimo per la risoluzione di sistemi lineari utilizzando il gradiente.

    La matrice è assunta simmetrica e definita positiva

    """
    def gradient(self) -> tuple[Any, timedelta, float, int]:
        # Per debugging
        if self.debug_gradient:
            self.prefix = '[DEBUG-G] '
        # Definisco il vettore iniziale nullo come inizio del metodo iterativo
        self.x = _get_start_vector(self.column)
        # Ottengo il tempo iniziale di esecuzione
        self._start_time()
        # Calcolo il residuo come r(k) = b − Ax(k)
        r = self.b - self.matrix.dot(self.x)
        # Calcolo alpha come r(k) trasposto ⋅ r(k) / r(k) trasposto ⋅ matrice ⋅ r(k) 
        alpha = (r.T.dot(r)) / (r.T @ self.matrix @ r)
        # Intero da i = 0 a iteration - 1
        for i in range(self.iteration):
            # Se il debug è attivo
            if self.debug_gradient:
                self._writeIteration(i)
            # Aggiorno il valore x: x(k+1) = x(k) + alpha(k) ⋅ r(k)
            self.x = self.x + alpha * r
            # Aggiorno r e alpha
            r = self.b - self.matrix.dot(self.x)
            alpha = (r.T.dot(r)) / (r.T @ self.matrix @ r)
            # Se arrivo alla tolleranza
            if self._isTolerate(self.x, self.b):
                if self.debug_gradient:
                    self._writeTolerance()
                # Ottengo il tempo finale di esecuzione
                self._end_time()
                i += 1  # Incremento per contare l'ultima iterazione
                return self.x, self._total_time(), self.relative_error(self.x), i
        # Se arrivo a max iterazioni
        if self.debug and not self.debug_gradient:
            self._writeSolution()
        # Ottengo il tempo finale di esecuzione
        self._end_time()
        return self.x, self._total_time(), self.relative_error(self.x), self.iteration

    """ 
    FUNZIONE:  conjugate_gradient(self) -> tuple[Any, timedelta, float, int]:

    DESCRIZIONE: 
    Funzione che implementa il Metodo del Gradiente Coniugato: x(k+1) = x(k) + alpha(k) ⋅ d(k)

    Il metodo del Gradiente Coniugato può essere visto come un miglioramento del metodo del Gradiente
    dove si va a risolvere il problema della convergenza a "zig-zag". Si vanno a cercare dei vettori 
    ottimali che non vengono più modificati lungo la direzione d.

    La matrice è assunta simmetrica e definita positiva

    """
    def conjugate_gradient(self) -> tuple[Any, timedelta, float, int]:
        # Per debugging
        if self.debug_conjugate_gradient:
            self.prefix = '[DEBUG-CG] '
        # Definisco il vettore iniziale nullo come inizio del metodo iterativo
        self.x = _get_start_vector(self.column)
        # Ottengo il tempo iniziale di esecuzione
        self._start_time()
        # Calcolo il residuo come r(k) = b − Ax(k)
        r = self.b - self.matrix.dot(self.x)
        # Assegno d = r
        d = r
        # Calcolo alpha come d(k) trasposto ⋅ r(k) / d(k) trasposto ⋅ matrice ⋅ d(k) 
        alpha = (d.T.dot(r)) / (d.T @ self.matrix @ d)
        # Intero da i = 0 a iteration - 1 
        for i in range(self.iteration):
            # Se debug è attivo
            if self.debug_conjugate_gradient:
                self._writeIteration(i)
            # Aggiorno il valore x: x(k+1) = x(k) + alpha(k) ⋅ d(k)
            self.x = self.x + alpha * d
            # Aggiorno il residuo
            r = self.b - self.matrix.dot(self.x)
            # Calcolo beta come d(k) trasposto ⋅ matrice ⋅ r(k+1) / d(k) trasposto ⋅ matrice ⋅ d(k) 
            beta = (d.T @ self.matrix @ r) / (d.T @ self.matrix @ d)
            # Aggiorno d(k+1) = r(k+1) - beta(k) * d(k)
            d = r - beta* d
            # Aggiorno alpha come d(k) trasposto ⋅ r(k) / d(k) trasposto ⋅ matrice ⋅ d(k)
            alpha = (d.T.dot(r)) / (d.T @ self.matrix @ d)
            # Se arrivo alla tolleranza
            if self._isTolerate(self.x, self.b):
                if self.debug_conjugate_gradient:
                    self._writeTolerance()
                # Ottengo il tempo finale di esecuzione
                self._end_time()
                i += 1
                return self.x, self._total_time(), self.relative_error(self.x), i
        # Se arrivo a max iterazioni
        if self.debug and not self.debug_conjugate_gradient:
            self._writeSolution()
        # Ottengo il tempo finale di esecuzione
        self._end_time()
        return self.x, self._total_time(), self.relative_error(self.x), self.iteration

    """ 
    FUNZIONE: _isTolerate(self, x, b):

    DESCRIZIONE: 
    Funzione che si occupa di ricalcolare il residuo e verificare
    se la norma del residuo / la norma di b è minore della tolleranza.
    Nel caso affermativo, è il criterio di arresto delle iterazioni

    """
    def _isTolerate(self, x, b):
        residual = self.matrix.dot(x) - b
        norm1 = np.linalg.norm(residual)
        norm2 = np.linalg.norm(b)
        return (norm1 / norm2) < self.tol

    """ 
    FUNZIONE: relative_error(self, x):

    DESCRIZIONE: 
    Funzione che si occupa di calcolare l'errore relativo tra la soluzione 
    ottenuta x e la soluzione esatta x' attraverso la formula:
    norma(x - x') / norma(x)

    """
    def relative_error(self, x):
        return np.linalg.norm(x - _get_solution(self.column)) / np.linalg.norm(x)

    """ 
    FUNZIONE:  diagonal_dominance(self):

    DESCRIZIONE: 
    Funzione che si occupa di verificare se una matrice è a dominanza diagonale

    """
    def diagonal_dominance(self):
        return self._row_diagonal_dominance() or self._column_diagonal_dominance()

    """ 
    FUNZIONE:  _row_diagonal_dominance(self):

    DESCRIZIONE: 
    Funzione che si occupa di verificare se la matrice utilizzata
    è a dominanza diagonale stretta per righe, ovvero se i valori sulla
    diagonale a(i,i) in abs di una matrice sono maggiori della somma dei abs
    dei valori della stessa riga escluso a(i,i)

    """
    def _row_diagonal_dominance(self):
        for i in range(self.matrix.shape[0]):
            tot = 0
            elem = abs(self.matrix[i, i])
            for j in range(self.row):
                tot += abs(self.matrix[i, j])
            if elem <= tot:
                return False
        return True

    """ 
    FUNZIONE:  _column_diagonal_dominance(self):

    DESCRIZIONE: 
    Funzione che si occupa di verificare se la matrice utilizzata
    è a dominanza diagonale stretta per colonne, ovvero se i valori sulla
    diagonale a(i,i) in abs di una matrice sono maggiori della somma dei abs
    dei valori della stessa colonna escluso a(i,i)

    """
    def _column_diagonal_dominance(self):
        for i in range(self.matrix.shape[0]):
            tot = 0
            elem = abs(self.matrix[i, i])
            for j in range(self.column):
                tot += abs(self.matrix[j, i])
            if elem <= tot:
                return False
        return True

    """ 
    FUNZIONE:  _getReverseP_jacobi(self):

    DESCRIZIONE: 
    Funzione che si occupa di calcolare P^(-1), dove P è la
    diagonale della matrice e P^(-1) la sua inversa, facilmente
    calcolabile dato che sarà composta dai reciproci degli 
    elementi sulla diagonale

    """
    def _getReverseP_jacobi(self):
        # Calcolo P^(-1)
        reverse_P = ss.diags([1 / self.matrix.diagonal()], [0])
        # Converto al formato sparso CSR nel caso non lo sia
        if not ss.isspmatrix_csr(reverse_P):
            reverse_P = reverse_P.tocsr()
        return reverse_P

    """ 
    FUNZIONE:  _getP_gauss(self):

    DESCRIZIONE: 
    Calcola la matrice P del metodi di Gauss Seidel, definita come la matrice triangolare
    inferiore della matrice matrix
    
    """
    def _getP_gauss(self):
        P = np.tril(self.matrix.toarray())
        return P
    
    """ 
    FUNZIONE:  _forward_substitution(self, P, residual):

    DESCRIZIONE: 
    Implementa l'algoritmo di sostituzione in avanti, al fine di evitare
    la computazione di P^-1 per l'implementazione di Gauss-Seidel
    
    """
    def _forward_substitution(self, P, residual):
        # Dimensione del sistema
        n = P.shape[0]
        # Inizializzazione a tutti zeri di x
        x = _get_start_vector(n)
        # Controllo se il primo termine è uguale a zero (determinante = 0)
        if P[0, 0] == 0:
            raise ValueError("[ERROR-GS] Il primo termine deve essere diverso da zero!")
        # Intero su n
        for i in range(n):
            # Se un elemento sulla diagonale è zero (determinante = 0)
            if P[i, i] == 0:
                raise ValueError("[ERROR-GS] La matrice è singolare, non può essere invertita.")   
            # Calcolo x(i) come r(i) - P(i,:)*x / P(i,i)
            x[i] = (residual[i] - P[i, :].dot(x)) / P[i, i]
        # Ritorno la soluzione della sostituzione in avanti
        return x

    """ 
    FUNZIONE DEBUG:  createLog(self):

    DESCRIZIONE: 
    Funzione che si occupa di creare dei file di log, utili
    per debugging

    """
    def createLog(self):
        # File latest.log
        file_name = 'latest.log'
        if os.path.exists(file_name):
            log_dir = 'log'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            old_log_name = 'latest.log'
            new_log_path = os.path.join(log_dir, f"log_{timestamp}.log")
            shutil.move(old_log_name, new_log_path)

    """ 
    FUNZIONE DEBUG:  write_to_log(self, message):

    DESCRIZIONE: 
    Funzione che si occupa di scrivere sul file "latest.log"

    """
    def write_to_log(self, message):
        with open('latest.log', 'a') as self.log:
            self.log.write(message)

    """ 
    FUNZIONE:  _start_time(self):

    DESCRIZIONE: 
    Funzione che si occupa di ottenere la data corrente.
    Utilizzata come tempo di riferimento iniziale per l'esecuzione
    di un metodo risolutivo
    
    """
    def _start_time(self):
        self.start = datetime.now()

    """ 
    FUNZIONE:  _end_time(self):

    DESCRIZIONE: 
    Funzione che si occupa di ottenere la data corrente.
    Utilizzata come tempo di riferimento finale per l'esecuzione
    di un metodo risolutivo
    
    """
    def _end_time(self):
        self.end = datetime.now()

    """ 
    FUNZIONE:  _total_time(self):

    DESCRIZIONE: 
    Funzione che si occupa di calcolare il tempo totale di 
    elaborazione di un metodo
    
    """
    def _total_time(self):
        return self.end - self.start

    """ 
    FUNZIONE:  _writeIteration(self, i):

    DESCRIZIONE: 
    Funzione che si occupa scrivere su log l'iterazione corrente di un metodo
    
    """
    def _writeIteration(self, i):
        self.write_to_log(self.prefix + "All'iterazione " + str(i) + ", x vale:\n")
        self.write_to_log('{}\n\n'.format(self.x))

    """ 
    FUNZIONE:  _writeTolerance(self):

    DESCRIZIONE: 
    Funzione che si occupa scrivere su log la tolleranza utilizzata
    
    """
    def _writeTolerance(self):
        self.write_to_log(self.prefix + 'Tolleranza a {}!\n'.format(self.tol))

    """ 
    FUNZIONE: _writeSolution(self):

    DESCRIZIONE: 
    Funzione che si occupa scrivere su log la soluzione ottenuta
    
    """
    def _writeSolution(self):
        self.write_to_log(self.prefix + 'Solution {}\n\n'.format(self.x))

""" 
FUNZIONE:  _get_start_vector(dim):

DESCRIZIONE: 
Funzione che si occupa di ritornare un array di dimensione dim di [0..0]
E' il vettore nullo

"""
def _get_start_vector(dim):
    return np.zeros(dim)

""" 
FUNZIONE: _get_solution(dim):

DESCRIZIONE: 
Funzione che si occupa di ritornare un array di dimensione dim di [1..1]
E' la soluzione esatta

"""
def _get_solution(dim):
    return np.ones(dim)

"""
    END OF FILE

"""