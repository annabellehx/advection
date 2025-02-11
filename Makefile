CC = mpicc
CFLAGS = -fopenmp -lm -O3

advection:
	${CC} advection_serial.c -o advection_serial ${CFLAGS}
	${CC} advection_shared.c -o advection_shared ${CFLAGS}
	${CC} advection_distributed.c -o advection_distributed ${CFLAGS}
	${CC} advection_hybrid.c -o advection_hybrid ${CFLAGS}

clean:
	rm -f advection_serial advection_shared advection_distributed advection_hybrid *.txt
