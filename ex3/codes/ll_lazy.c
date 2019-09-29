#include <stdio.h>
#include <stdlib.h> /* rand() */
#include <limits.h>
#include <pthread.h> /* for pthread_spinlock_t */

#include "alloc.h"
#include "ll.h"

typedef struct ll_node {
	int key;
	int tag;
	struct ll_node *next;
	
	pthread_spinlock_t state ;

} ll_node_t;

struct linked_list {
	ll_node_t *head;
};

/**
 * Create a new linked list node.
 **/
static ll_node_t *ll_node_new(int key)
{
	ll_node_t *ret;

	XMALLOC(ret, 1);
	ret->key = key;
	ret->next = NULL;
	ret->tag = 0;
	pthread_spin_init(&ret->state,PTHREAD_PROCESS_PRIVATE);

	return ret;
}

/**
 * Free a linked list node.
 **/
static void ll_node_free(ll_node_t *ll_node)
{
	pthread_spin_destroy(&ll_node->state);
	XFREE(ll_node);
}

/**
 * Create a new empty linked list.
 **/
ll_t *ll_new()
{
	ll_t *ret;

	XMALLOC(ret, 1);
	ret->head = ll_node_new(-1);
	ret->head->next = ll_node_new(INT_MAX);
	ret->head->next->next = NULL;

	return ret;
}

/**
 * Free a linked list and all its contained nodes.
 **/
void ll_free(ll_t *ll)
{
	ll_node_t *next, *curr = ll->head;
	while (curr) {
		next = curr->next;
		ll_node_free(curr);
		curr = next;
	}
	XFREE(ll);
}

int ll_contains(ll_t *ll, int key)
{
	int ret = 0;
	ll_node_t *curr;
	curr=ll->head ;
	//curr=curr->next;

	while(curr->key < key) {
		curr=curr->next;
	}

	ret = ((key==curr->key) && (!curr->tag));
	return ret;
}


int validate(ll_node_t *prev, ll_node_t *curr)
{
	int ret = 0;

	ret = ((!prev->tag) && (!curr->tag) && (prev->next==curr));
        return ret;
}

int ll_add(ll_t *ll, int key)
{
    	int ret = 0;
    	ll_node_t *new_node;
    	ll_node_t *prev,*curr;

    	while(1) {

   	 	prev = ll->head;
   	 	curr = prev->next;

   	 	while( curr->key <= key) {
		 	if(curr->key == key) break;
   		 	prev = curr;
   		 	curr = curr->next;
   	 	}
   	 	pthread_spin_lock(&(prev->state));
   	 	pthread_spin_lock(&(curr->state));

   	 	if(validate(prev,curr)){
   		 	if(curr->key != key){
   			 	new_node=ll_node_new(key);
   			 	new_node->next = curr;
   			 	prev->next = new_node;
   			 	ret = 1;
   		 	}
   		 	pthread_spin_unlock(&(prev->state));
   		 	pthread_spin_unlock(&(curr->state));
   		 	break;
   	 	}
   	 	pthread_spin_unlock(&(prev->state));
   	 	pthread_spin_unlock(&(curr->state));
    	}
    	return ret;
}


int ll_remove(ll_t *ll, int key)
{

    	int ret=0;
    	ll_node_t *prev,*curr;

    	while(1) {

   	 	prev=ll->head;
   	 	curr=prev->next;

   	 	while(curr->key <= key) {
   		 	if(key == curr->key) break;
   		 	prev=curr;
   		 	curr=curr->next;
   	 	}

   	 	pthread_spin_lock(&(prev->state));
   	 	pthread_spin_lock(&(curr->state));

   	 	if(validate(prev,curr)) {
   		 	if(curr->key == key) {
				curr->tag = 1;
   			 	prev->next=curr->next;
   			 	ret=1;
			}
   			pthread_spin_unlock(&(prev->state));
   			pthread_spin_unlock(&(curr->state));
   			break;
   	 	}
   	 	pthread_spin_unlock(&(prev->state));
   	 	pthread_spin_unlock(&(curr->state));
    	}

    	return ret;
}
/**
 * Print a linked list.
 **/
void ll_print(ll_t *ll)
{
	ll_node_t *curr = ll->head;
	printf("LIST [");
	while (curr) {
		if (curr->key == INT_MAX)
			printf(" -> MAX");
		else
			printf(" -> %d", curr->key);
		curr = curr->next;
	}
	printf(" ]\n");
}
