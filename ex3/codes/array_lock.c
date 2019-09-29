#include "lock.h"
#include "alloc.h"
#include <pthread.h>

static __thread int index;

struct lock_struct {

	int* flag;
	int tail, size;
	
};

lock_t *lock_init(int nthreads)
{
	lock_t *lock;

	XMALLOC(lock, 1);
	/* other initializations here. */
	lock->size = nthreads;
	XMALLOC(lock->flag, lock->size);
	lock->flag[0] = 1;
	lock->tail = 0; 
	return lock;
}

void lock_free(lock_t *lock)
{	
	XFREE(lock->flag);
	XFREE(lock);
	return;
}

void lock_acquire(lock_t *lock)
{
	index = __sync_fetch_and_add(&(lock->tail),1)%(lock->size);
	while(!lock->flag[index]) ;
	return;

}

void lock_release(lock_t *lock)
{
	lock->flag[index] = 0;
	lock->flag[(index+1) % (lock->size)] = 1;
	return;

}
