CC=gcc
LIBS=-lglut -lGLU -lGL -lm
INCLUDES=-I/usr/include/GL
CCOPTS=	-Wall -03	-std=gnu99	$(INCLUDES)

main.o:	main.c	mmu.h	
	gcc	-std=gnu99	-o	main	main.c	mmu.c -g

clean:
	rm -f main
