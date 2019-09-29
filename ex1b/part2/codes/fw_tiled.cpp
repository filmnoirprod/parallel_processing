/*
  Tiled version of the Floyd-Warshall algorithm.
  command-line arguments: N, B
  N = size of graph
  B = size of tile
  works only when N is a multiple of B
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "util.h"
#include <emmintrin.h>
#include "tbb/task.h"
#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include <immintrin.h>

using namespace std;


inline int min(int a, int b);
inline void FW(int **A, int K, int I, int J, int N);

int main(int argc, char **argv) {
	int **A;
     	int i,k,nthreads;
     	struct timeval t1, t2;
     	double time;
     	int B=64;
     	int N=1024;

     	if (argc != 4){
        	fprintf(stdout, "Usage %s N B nthreads\n", argv[0]);
        	exit(0);
     	}

     	N=atoi(argv[1]);
     	B=atoi(argv[2]);
     	nthreads=atoi(argv[3]);

     	tbb::task_scheduler_init init(nthreads);

     	A=(int **)malloc(N*sizeof(int *));
     	for(i=0; i<N; i++)A[i]=(int *)malloc(N*sizeof(int));

     	graph_init_random(A,-1,N,128*N);

        tbb::task_group g;

     	gettimeofday(&t1,0);


     	for(k=0;k<N;k+=B){
	
        	g.run( [=]{ FW(A,k,k,k,B); } );

        	g.wait();

        	for(i=0; i<N; i+=B) {

			if(i==k) continue ;
          		g.run( [=] { FW(A,k,i,k,B);
	  		FW(A,k,k,i,B); } );

		}

 


		g.wait();
	
		for(i=0; i<N; i+=B) {
		
			if(i==k) continue ;

	 		g.run( [=] { 

			for(int j=0; j<N; j+=B) {

				if( (i != k) && ( j != k ) ) {
                       
					FW(A,k,i,j,B); 
				} 
			}
			}); 
		}
		   
 

		g.wait();

     }
 
     gettimeofday(&t2,0);

     time=(double)((t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec)/1000000;
     printf("FW_TILED,%d,%d,%.4f\n", N,B,time);
 //    printf("sizeofint= %lu\n",sizeof(int));
     
    /* for(i=0; i<N; i++)
        for(j=0; j<N; j++) fprintf(stdout,"%d\n", A[i][j]);*/
     

     return 0;
}

inline int min(int a , int b) {

	if(a<=b) return a;
	else return b;
}


inline void FW(int **A, int K, int I, int J, int N) {

	int i,j,k;

	__m128i *p;
	__m128i comp,Aij,Akj,Aik,Ai1k,Ai2k,Ai3k;


	for( k=K ; k<K+N ; k++) {

		for( i=I ; i<I+N ; i+=4) {
			
			Aik=_mm_set1_epi32(A[i][k]);
                        Ai1k=_mm_set1_epi32(A[i+1][k]);
                        Ai2k=_mm_set1_epi32(A[i+2][k]);
                        Ai3k=_mm_set1_epi32(A[i+3][k]);

	   		for ( int j=J ; j<J+N ; j+=4) { 
					

                                        Akj=_mm_load_si128((__m128i*) &A[k][j]);

					if(i!=k) {

           					Aij=_mm_load_si128((__m128i*) &A[i][j]);
						comp =_mm_add_epi32(Aik,Akj);
						Aij =_mm_min_epi32(Aij,comp);
		                             
						_mm_store_si128((__m128i*) &A[i][j],Aij);
					}

                                        Aij=_mm_load_si128((__m128i*) &A[i+1][j]);
                                        comp =_mm_add_epi32(Ai1k,Akj);
                                        Aij =_mm_min_epi32(Aij,comp);

                                        _mm_store_si128((__m128i*) &A[i+1][j],Aij);

                                        Aij=_mm_load_si128((__m128i*) &A[i+2][j]);
                                        comp =_mm_add_epi32(Ai2k,Akj);
                                        Aij =_mm_min_epi32(Aij,comp);

                                        _mm_store_si128((__m128i*) &A[i+2][j],Aij);

                                        Aij=_mm_load_si128((__m128i*) &A[i+3][j]);
                                        comp =_mm_add_epi32(Ai3k,Akj);
                                        Aij =_mm_min_epi32(Aij,comp);

                                        _mm_store_si128((__m128i*) &A[i+3][j],Aij);

		

				}
	 		}

	}



}
