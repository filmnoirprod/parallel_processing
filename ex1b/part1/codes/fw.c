/*
 Standard implementation of the Floyd-Warshall Algorithm
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "util.h"
#include <omp.h>
#include <immintrin.h>
#include <emmintrin.h>


inline int min(int a, int b);

int main(int argc, char **argv)
{
     int **A;
     int i,j,k;
     struct timeval t1, t2;
     double time;
     int N=1024;

     if (argc != 2) {
        fprintf(stdout,"Usage: %s N\n", argv[0]);
        exit(0);
     }

     N=atoi(argv[1]);
    // t=atoi(argv[2]);

     A = (int **) malloc(N*sizeof(int *));
     for(i=0; i<N; i++) A[i] = (int *) malloc(N*sizeof(int));

     graph_init_random(A,-1,N,128*N);

     gettimeofday(&t1,0);
     __m128i *p;
        //__m128i  *p1;
     __m128i comp,Aij,Akj,Aik,Ai1k,Ai2k,Ai3k,Mask;

     for(k=0;k<N;k++) {
	
	#pragma omp parallel for  private(i)
	for(i=0; i< N ; i+=4) {
			Aik=_mm_set1_epi32(A[i][k]);
                        Ai1k=_mm_set1_epi32(A[i+1][k]);
                        Ai2k=_mm_set1_epi32(A[i+2][k]);
                        Ai3k=_mm_set1_epi32(A[i+3][k]);
			#pragma omp parallel for private(j)
	   		for ( j=0 ; j<N ; j+=4) { 
                                        Akj=_mm_load_si128((__m128i*) &A[k][j]);

					if(i!=k) {

						p=(__m128i*)(&A[i][j]);		
           					Aij=_mm_load_si128(p);
						comp =_mm_add_epi32(Aik,Akj);
						Aij =_mm_min_epi32(Aij,comp);
	                                        //Mask=_mm_cmplt_epi32(Aij,comp);          
        	                                //Aij=_mm_or_si128( _mm_and_si128(Mask,Aij), _mm_andnot_si128(Mask,comp) );


		                             
						_mm_store_si128(p,Aij);
					}

                                        p=(__m128i*)(&A[i+1][j]);
                                        Aij=_mm_load_si128(p);
                                        comp =_mm_add_epi32(Ai1k,Akj);
                                        Aij =_mm_min_epi32(Aij,comp);
                                        //Mask=_mm_cmplt_epi32(Aij,comp);          
                                        //Aij=_mm_or_si128( _mm_and_si128(Mask,Aij), _mm_andnot_si128(Mask,comp) );


                                        _mm_store_si128(p,Aij);

                                        p=(__m128i*)(&A[i+2][j]);
                                        Aij=_mm_load_si128(p);
                                        comp =_mm_add_epi32(Ai2k,Akj);
                                        Aij =_mm_min_epi32(Aij,comp);
                                        //Mask=_mm_cmplt_epi32(Aij,comp);          
                                        //Aij=_mm_or_si128( _mm_and_si128(Mask,Aij), _mm_andnot_si128(Mask,comp) );


                                        _mm_store_si128(p,Aij);

                                        p=(__m128i*)(&A[i+3][j]);
                                        Aij=_mm_load_si128(p);
                                        comp =_mm_add_epi32(Ai3k,Akj);
                                        Aij =_mm_min_epi32(Aij,comp);

					//Mask=_mm_cmplt_epi32(Aij,comp);        
					//Aij=_mm_or_si128( _mm_and_si128(Mask,Aij), _mm_andnot_si128(Mask,comp) );


                                        _mm_store_si128(p,Aij);

		

				}
	 		}
		}	


	



/*

	for(k=K; k<K+N; k++)
		for(i=I; i<I+N; i++)
			for(j=J; j<J+N; j++) {

				A[i][j]=min(A[i][j],A[i][k]+A[k][j]);

			} 

*/









	
	/*	if ( i==k ) continue ;

		for(j=0; j<N ; j++)
			A[i][j]=min(A[i][j], A[i][k]+A[k][j]);


	}

    }*/

      /* for(i=0; i<N; i++)
           for(j=0; j<N; j++)
              A[i][j]=min(A[i][j], A[i][k] + A[k][j]);
	*/
     gettimeofday(&t2,0);

     time=(double)((t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec)/1000000;
     printf("FW,%d,%.4f\n", N, time);

     /*
     for(i=0; i<N; i++)
        for(j=0; j<N; j++) fprintf(stdout,"%d\n", A[i][j]);
     */

     return 0;     
}

inline int min(int a, int b)
{
     if(a<=b)return a;
     else return b;
}

