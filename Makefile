CC = mpicc
CFLAGS = -fopenmp -lm -O3

advection:
	${CC} advection_shared.c -o advection_serial ${CFLAGS}
	${CC} advection_shared.c -o advection_shared -DUSE_OMP ${CFLAGS}
	${CC} advection_distributed.c -o advection_distributed ${CFLAGS}
	${CC} advection_distributed.c -o advection_hybrid -DUSE_OMP ${CFLAGS}

clean:
	rm -f advection_serial advection_shared advection_distributed advection_hybrid *.txt
