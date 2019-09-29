#include "lock.h"
#include "alloc.h"
#include <pthread.h>

struct lock_struct {
	pthread_spinlock_t mylock ;
};

lock_t *lock_init(int nthreads)
{
	lock_t *lock;

	XMALLOC(lock, 1);
	/* other initializations here. */
	pthread_spin_init(&lock->mylock,1);
	return lock;
}

void lock_free(lock_t *lock)
{
	XFREE(lock);
}

void lock_acquire(lock_t *lock)
{
	pthread_spin_lock(&lock->mylock);
}

void lock_release(lock_t *lock)
{
	pthread_spin_unlock(&lock->mylock);
}
