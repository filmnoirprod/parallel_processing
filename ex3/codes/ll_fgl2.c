#include <stdio.h>
#include <stdlib.h> /* rand() */
#include <limits.h>
#include <pthread.h> /* for pthread_spinlock_t */

#include "alloc.h"
#include "ll.h"

typedef struct ll_node {
	int key;
	pthread_spinlock_t state;
	struct ll_node *next;

	/* other fields here? */
} ll_node_t;

struct linked_list {
	ll_node_t *head;
	/* other fields here? */
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
	/* Other initializations here? */
	int pshared;
	pthread_spin_init(&ret->state,pshared);
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
	//printf("Contains \n");

	ll_node_t *prev,*curr;
        pthread_spin_lock(&ll->head->state);
        prev=ll->head;
        curr = prev->next;
        pthread_spin_lock(&curr->state);
        while(curr->key<=key) {
                        if(curr->key==key) {
                                pthread_spin_unlock(&prev->state);
                                pthread_spin_unlock(&curr->state);                        
                                return 1; 
                        }
                        pthread_spin_unlock(&prev->state);
                        prev=curr;
                       // if(curr->next==NULL) break;
                        curr = curr->next;
                        pthread_spin_lock(&curr->state);
                }
               /* if(curr->next!=NULL)*/ pthread_spin_unlock(&curr->state);
                pthread_spin_unlock(&prev->state);
        
        
                        
        return 0;}

int ll_add(ll_t *ll, int key)
{
	ll_node_t *prev,*curr;
        pthread_spin_lock(&ll->head->state);
	prev = ll->head;
        if(prev->key>key) {
                ll->head = ll_node_new(key);
                ll->head->next=prev;
                pthread_spin_unlock(&prev->state);
                 }
        else if(prev->next==NULL) {
                curr = ll_node_new(key);
                prev->next=curr;
                pthread_spin_unlock(&prev->state);
                
        }
	else {
        curr=prev->next;
        pthread_spin_lock(&curr->state);
        while(curr->key<key) {
                pthread_spin_unlock(&prev->state);
                prev = curr;
                curr = curr->next;
                if(curr==NULL) break;
                pthread_spin_lock(&curr->state);
                }
        prev->next=ll_node_new(key);
        prev->next->next=curr;
        pthread_spin_unlock(&prev->state);        
        if(curr!=NULL) pthread_spin_unlock(&curr->state);
}
        return key;
}

int ll_remove(ll_t *ll, int key)
{
//	printf("Remove\n");
	int ret=0;
	ll_node_t *prev,*curr;
        pthread_spin_lock(&ll->head->state);
	prev = ll->head;
        curr = prev->next;
        pthread_spin_lock(&curr->state);
        /*if(prev->key==key) {
                ll->head=curr;
                pthread_spin_unlock(&prev->state);
                pthread_spin_unlock(&curr->state);
                ret =  key;
        }
	else {  */     
        while (curr->key <= key) {
                if(curr->key==key) {    
			prev->next=curr->next;
                        pthread_spin_unlock(&prev->state);
                        ll_node_free(curr);
                        ret= 1; 
			break;}
		else {
                pthread_spin_unlock(&prev->state);
                prev = curr;
                //if(curr->next==NULL) break;
                curr = curr->next;
                pthread_spin_lock(&curr->state);
        }}
	if(ret==0) {
        pthread_spin_unlock(&prev->state);
        /*if(prev!=curr)*/ pthread_spin_unlock(&curr->state);}
        
         
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

