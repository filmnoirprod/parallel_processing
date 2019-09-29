#include <stdio.h>
#include <stdlib.h> /* rand() */
#include <limits.h>
#include <pthread.h> /* for pthread_spinlock_t */

#include "alloc.h"
#include "ll.h"

typedef struct ll_node {
	int key;
	struct ll_node *next;
	char padding2[64-sizeof(int)-sizeof(struct ll_node*)];

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

	return ret;
}

/**
 * Free a linked list node.
 **/
static void ll_node_free(ll_node_t *ll_node)
{
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

	ret = ((key==curr->key) && (!(long long)curr)&1);
	return ret;
}

int find(ll_t *ll, int key, ll_node_t **prev, ll_node_t **curr) {

	int snip = 0;
	int flag = 0;
	ll_node_t *previous,*current;

	while(1) {
		
		snip=0;
		previous = ll->head;
		current = previous->next;

		while(1) {

			if(((long long)current)&1) {
				snip = __sync_bool_compare_and_swap(&previous->next,current,current->next);
				break;
			}

			if(current->key >= key) {
				flag=1;
				break;
			}
			previous = current;
			current = current->next;
		}
		if(!snip && !flag) continue;
		*prev = previous;
		*curr = current;
		return 1;
	}
}


int ll_add(ll_t *ll, int key)
{
    	int ret = 0;
	int snip = 0;
    	ll_node_t *new_node;
    	ll_node_t *prev,*curr;

    	while(1) {

		find(ll,key,&prev,&curr);
		
		if(curr->key == key) break;

		new_node = ll_node_new(key);
		__sync_bool_compare_and_swap(&new_node->next,new_node->next,curr);
		snip = __sync_bool_compare_and_swap(&prev->next,curr,new_node);
		if(!snip) break;
		ret = 1;
		break;
	}
   	 
    	return ret;
}


int ll_remove(ll_t *ll, int key)
{

    	int ret=0;
	int snip=0;
    	ll_node_t *prev,*curr;

    	while(1) {

   	 	find(ll,key,&prev,&curr);
		
		if(curr->key!=key) break;
		
		snip = __sync_bool_compare_and_swap(&curr,(long long)curr&(~1),(long long)curr|1);
		if(!snip) continue;
		__sync_bool_compare_and_swap(&prev->next,curr,curr->next);
		ret=1;
		break;
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
