"""
    FILE MAIN.PY 

    AUTORI: 
    - Biancini Mattia 865966
    - Gargiulo Elio 869184

    DESCRIZIONE:
    File principale del progetto. 
    Contiene le funzioni necessarie per caricare le matrici della cartella "matrix" 
    in memoria, per poi essere utilizzate per la risoluzione tramite i quattro metodi
    implementati in "methods.py"

"""
# Per numeri casuali
from random import randrange
# Per operazioni
import numpy as np
# Libreria dei metodi risolutivi
from methods import LinearSystemSolver
# Per l'import e gestione in memoria di matrici
import scipy as sp
# Per directories
import os
from datetime import datetime

""" 
    FUNZIONE: load_matrices():

    DESCRIZIONE:
    Carica le matrici sparse contenute nella cartella "./matrix" 
    nel progetto e le ritorna al chiamante

"""
def load_matrices():
    # Ottengo la directory
    matrix_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_dir = os.path.join(matrix_dir, 'matrix')
    # Ottengo i file contenuti nella directory
    matrix_files = os.listdir(matrix_dir)
    matrices = []
    for x in matrix_files:
        # Se i file sono effettivamente matrici "Matrix Market"
        if x.__contains__('.mtx'):
            # Appendo le matrici
            matrices.append(os.path.join(matrix_dir, x))

    return matrices

""" 
    FUNZIONE: import_random_matrix(matrices):

    DESCRIZIONE:
    Se la lista di matrici "matrices" non è vuota ritorna
    una matrice casuale della lista al chiamante

"""
def import_random_matrix(matrices):
    # Se la lunghezza della lista è 0 (vuota)
    if len(matrices) == 0:
        return
    # Seleziono una matrice casuale e la assegno a matrix
    matrix = sp.io.mmread(matrices[randrange(len(matrices))])
    return matrix

""" 
    FUNZIONE: import_matrix(matrices, index):

    DESCRIZIONE:
    Variante della funzione "import_random_matrix(matrices) che permette
    la selezione di una specifica matrice in posizione "index", nel caso sia
    un indice corretto. Altrimenti ritorna al chiamante una matrice casuale

"""
def import_matrix(matrices, index):
    # Se la lunghezza della lista è 0 (vuota)
    if len(matrices) == 0:
        return
    # Se l'indice non è valido
    if index >= len(matrices) or index < 0:
        print('This is invalid number ({}). A random Matrix will be choose.'.format(index))
        return import_random_matrix(matrices)
    # Seleziono una matrice casuale e la assegno a matrix
    matrix = sp.io.mmread(matrices[index])
    return matrix

""" 
    FUNZIONE: get_random_matrix():

    DESCRIZIONE:
    Funzione principale che incapsula "import_random_matrix" con "load_matrices"
    parametro, al fine di importare e selezionare una matrice casuale con un unica 
    chiamata.

"""
def get_random_matrix():
    return import_random_matrix(load_matrices())

""" 
    FUNZIONE: get_matrix(index):

    DESCRIZIONE:
    Funzione principale che incapsula "import_matrix" con "load_matrices"
    parametro, al fine di importare e selezionare una matrice dato un indice
    "index" con un unica chiamata.

"""
def get_matrix(index):
    return import_matrix(load_matrices(), index)

""" 
    FUNZIONE: write_results_to_file(data, method, tolerance, matrix_number):

    DESCRIZIONE:
    Funzione che permette la scrittura su un file "risultati.txt" di un'intera esecuzione
    del main. In particolare vengono scritti su file:
    - data: tupla contentente i seguenti dati:
            - soluzione ottenuta
            - tempo impiegato
            - errore relativo
            - numero di iterazioni
    - method: nome del metodo utilizzato per la risoluzione del sistema
    - tolerance: valore di tolleranza usata per arrestare le iterazioni (convergenza)
    - matrix_number: numero della matrice trattata

"""
def write_results_to_file(data, method, tolerance, matrix_number):
    # Verifica se il file "risultati.txt" esiste già
    file_name = "risultati.txt"
    file_exists = os.path.exists(file_name)
    # Se il file non esiste, crea un nuovo file "risultati.txt"
    if not file_exists:
        with open(file_name, "w") as file:
            file.write("====[ DATI ESECUZIONE METODI ]====\n\n")
    # Scrivi i dati nella tupla nel file
    with open(file_name, "a") as file:
        file.write('=== {} Method ===\n'.format(method))
        file.write('Matrice: {}\n'.format(matrix_number))
        file.write('Tempo Impiegato: {:.9f}\n'.format(data[1].total_seconds()))
        file.write('Errore Relativo: {}\n'.format(data[2]))
        file.write('Numero di Iterazioni {}/50000\n'.format(data[3]))
        file.write('Tolleranza: {}\n'.format(tolerance))
        file.write('X = {}\n\n'.format(data[0]))

def write_format_to_file(data, method, tolerance, matrix_number):
    # Verifica se il file "risultati.txt" esiste già
    file_name = "format.txt"
    file_exists = os.path.exists(file_name)
    # Se il file non esiste, crea un nuovo file "risultati.txt"
    if not file_exists:
        with open(file_name, "w") as file:
            file.write("====[ DATI ESECUZIONE METODI ]====\n\n")
    # Scrivi i dati nella tupla nel file
    with open(file_name, "a") as file:
        file.write('=== {} Method ===\n'.format(method))
        file.write('Matrice: {}\n'.format(matrix_number))
        file.write('Tempo Impiegato: {:.9f}\n'.format(data[1].total_seconds()))
        file.write('Errore Relativo: {}\n'.format(data[2]))
        file.write('Numero di Iterazioni {}/50000\n'.format(data[3]))
        file.write('Tolleranza: {}\n\n'.format(tolerance))

""" 
    FUNZIONE: __main__:

    DESCRIZIONE:
    Funzione dove inizia la vera e propria esecuzione del programma.
    E' presente una prima parte concentrata al debugging dove vengono
    provati, se "solve_all = True", tutti i metodi implementati della libreria
    in "methods.py", con risultati e informazioni scritti su file "risultati.txt"
    La seconda parte è un esempio di programma dove attraverso la console si 
    possono specificare parametri (matrice, tolleranza e metodo di risoluzione)
    per poi ottenere i risultati desiderati


"""
if __name__ == "__main__":
    # Inizializzo variabili e opzioni di stampa per soluzione
    debug = True
    solve_all = True
    np.set_printoptions(threshold=np.inf)
    maxIteration = 50000
    # Se il debug è attivo
    if debug:
        # Scrivo sul file tutti i risultati per ogni esecuzione di ogni metodo
        if solve_all:
            # Carico le matrici
            matrices = load_matrices()
            tol = [1e-4, 1e-6, 1e-8, 1e-10]  # Lista di tolleranze per scrittura file
            for i in range(len(matrices)):
                for j in range(4):
                    # Estraggo la matrice i
                    matrix = import_matrix(matrices, i)
                    # Definisco il LinearSystemSolver, il costruttore offerto dalla
                    # libreria implementata in methods.py
                    solver = LinearSystemSolver(matrix, maxIteration, False, tol_index=j)
                    result = solver.jacobi()  # Metodo di Jacobi
                    write_results_to_file(result, 'Jacobi', tol[j], i)
                    write_format_to_file(result, 'Jacobi', tol[j], i)
                    result = solver.gauss_seidel()  # Metodo di Gauss-Seidel
                    write_results_to_file(result, 'Gauss-Seidel', tol[j], i)
                    write_format_to_file(result, 'Gauss-Seidel', tol[j], i)
                    result = solver.gradient()  # Metodo del Gradiente
                    write_results_to_file(result, 'Gradient', tol[j], i)
                    write_format_to_file(result, 'Gradient', tol[j], i)
                    result = solver.conjugate_gradient()  # Metodo del Gradiente Coniugato
                    write_results_to_file(result, 'Conjugate Gradient', tol[j], i)
                    write_format_to_file(result, 'Conjugate Gradient', tol[j], i)
        # Altrimenti testo il metodo che necessita di debugging
        else:
            # Estraggo una matrice
            matrix = get_matrix(2)
            # Solver e metodo
            solver = LinearSystemSolver(matrix, maxIteration, debug)
            result = solver.gauss_seidel()
            # Stampe dei risultati
            print('La Soluzione calcolata è: ', result[0])
            print('Tempo Impiegato: ', result[1])
            print('Errore Relativo: ', result[2])
    # Altrimenti procedo con la normale esecuzione del main
    else:
        # Selezione di una matrice specifica (da 1 a 4)
        index = int(input('Scegli la matrice (Un numero da 1 a 4): '))
        matrix = get_matrix(index)
        # Scelta della tolleranza in notazione scientifica
        tol = float(input('Inserire la tolleranza (usa la notazione 1e-10): '))
        # Scelta del metodo di risoluzione del sistema
        print('Scegli uno dei seguenti metodi risolutivi: ')
        print('1) Jacobi')
        print('2) Gauss-Seidel')
        print('3) Gradiente')
        print('4) Gradiente Coniugato')
        method = int(input('Numero del metodo: '))
        # Costruzione del risolutore e applicazione del metodo in base alla scelta
        solver = LinearSystemSolver(matrix, maxIteration, False, tol=tol)
        if method == 1:
            result = solver.jacobi()
        elif method == 2:
            result = solver.gauss_seidel()
        elif method == 3:
            result = solver.gradient()
        else:
            result = solver.conjugate_gradient()
        # Stampa i risultati ottenuti dalla risoluzione
        print('Tempo Impiegato: ', result[1])
        print('Errore Relativo: ', result[2])
        if result[3] == maxIteration:
            print('Numero massimo di iterazioni raggiunto ({})'.format(maxIteration))
        else:
            print('Tolleranza di ', tol, ' soddisfatta dopo ', result[3], '/', maxIteration, ' iterazioni.')
        print('La soluzione calcolata è: ', result[0])

"""
    END OF FILE

"""