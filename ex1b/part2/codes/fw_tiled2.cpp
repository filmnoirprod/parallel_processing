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
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include <immintrin.h>

using namespace std;


inline int min(int a, int b);
inline void FW(int **A, int K, int I, int J, int N);

int main(int argc, char **argv) {
	int **A;
     	int i,k,nthreads,end;
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
	tbb::affinity_partitioner ap;
     	gettimeofday(&t1,0);

     	//tbb::task_group g;
	
     	for(k=0;k<N;k+=B){
	
        	g.run( [=]{ FW(A,k,k,k,B); } );

        	g.wait();
		end=(N/B);		
		tbb::parallel_for(
		tbb::blocked_range<int>(0,end), [=](const tbb::blocked_range<int>& r) {
        	for(int i=r.begin() ,i_end=r.end() ,i1=i*B ; i<i_end; i++) {
			if((i1!=k)) {
				FW(A,k,i1,k,B);
				FW(A,k,k,i1,B);
			}

			
		}
		},
		ap );
	
 			end=(N/(B));
			tbb::parallel_for(
			tbb::blocked_range2d<int>(0,end,0,end) , [=](const tbb::blocked_range2d<int>& t) {
			for(int i=t.rows().begin(),iend=t.rows().end(),i2=i*B; i<iend ; i++) {
				for(int j=t.cols().begin(),j_end=t.cols().end(),j2=j*B; j<j_end; j++) {

					if( (i2 != k) && ( j2 != k ) ) {
                       
						FW(A,k,i2,j2,B); 
					} 
				}
			}
			},
			ap);
		   

     }
 
     gettimeofday(&t2,0);

     time=(double)((t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec)/1000000;
     printf("FW_TILED,%d,%d,%.4f\n", N,B,time);
 //    printf("sizeofint= %lu\n",sizeof(int));
     
     //for(i=0; i<N; i++)
        //for(int j=0; j<N; j++) fprintf(stdout,"%d\n", A[i][j]);
     

     return 0;
}

inline int min(int a , int b) {

	if(a<=b) return a;
	else return b;
}


inline void FW(int **A, int K, int I, int J, int N) {

	int i,j,k;

	__m128i *p;
	//__m128i  *p1;
	__m128i comp,Aij,Akj,Aik,Ai1k,Ai2k,Ai3k,Mask;
//        __m128i  *p2=(__m128i*)(&temp);


	for( k=K ; k<K+N ; k++) {

		for( i=I ; i<I+N ; i+=4) {
			
                	//__m128i Aik=_mm_load_si128(p2);sa
			Aik=_mm_set1_epi32(A[i][k]);
                        Ai1k=_mm_set1_epi32(A[i+1][k]);
                        Ai2k=_mm_set1_epi32(A[i+2][k]);
                        Ai3k=_mm_set1_epi32(A[i+3][k]);

	   		for ( int j=J ; j<J+N ; j+=4) { 
					

					//p1=(__m128i*)(&A[k][j]);
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
	//	}	


	}



/*

	for(k=K; k<K+N; k++)
		for(i=I; i<I+N; i++)
			for(j=J; j<J+N; j++) {

				A[i][j]=min(A[i][j],A[i][k]+A[k][j]);

			} 

*/


}
