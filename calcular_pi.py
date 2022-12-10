# El método gather hace lo contrario al scatther. Si cada proceso tiene un elemento, 
# se puede usar gather para recopilarlos en una lista de elementos en el proceso maestro. El siguiente 
# código de ejemplo usa scatther y gatther para calcular π en paralelo.


from mpi4py import MPI
import time
import math

t0 = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Numero de pasos de integracion
n_pasos = 1000000000
dx = 1.0 / n_pasos

if rank == 0:
    # determina el tamaño de cada subtarea
    ave, res = divmod(n_pasos, nprocs)
    counts = [ave + 1 if p < res else ave for p in range(nprocs)]

    # determina los indices de inicio y finalizacion de cada subtarea
    inicio = [sum(counts[:p]) for p in range(nprocs)]
    final = [sum(counts[:p+1]) for p in range(nprocs)]

    # guardar los índices inicial y final en datos
    data = [(inicio[p], final[p]) for p in range(nprocs)]
else:
    data = None

data = comm.scatter(data, root=0)

# calcular la distribucion parcial a pi en cada subproceso
partial_pi = 0.0
for i in range(data[0], data[1]):
    x = (i + 0.5) * dx
    partial_pi += 4.0 / (1.0 + x * x)
partial_pi *= dx

partial_pi = comm.gather(partial_pi, root=0)

if rank == 0:
    print('Calculo de pi en {:.3f} sec'.format(time.time() - t0))
    print('error de {}'.format(abs(sum(partial_pi) - math.pi)))