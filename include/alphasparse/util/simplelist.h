
#pragma once
#include <stdlib.h>
/**
 *  \file oski/simplelist.h
 *  \brief Simple read-only, lockable, append-only list implementation.
 **/

/**
 *  \brief Node in the linked list.
 *
 *  A node encapsulates an arbitrary data element, and furthermore
 *  contains a pointer to the next element of the list (or NULL if
 *  none).
 *
 *  Note that the element should logically be read-only.
 */
typedef struct tagSimplenode_t
{
    const void *element;          /**< Linked-list node data. */
    struct tagSimplenode_t *next; /**< Next element. */
} simplenode_t;

/**
 *  \brief Simple list object.
 */
typedef struct tagSimplelist_t
{
    simplenode_t *head;  /**< Pointer to the head node. */
    simplenode_t *tail;  /**< Pointer to the tail node. */
    size_t num_elements; /**< Number of elements in the list. */
} simplelist_t;

/**
 *  \brief Returns a newly allocated list object.
 */
simplelist_t *simplelist_Create(void);

/**
 *  \brief Deallocates memory associated with a simple list.
 */
void simplelist_Destroy(simplelist_t *L);

/**
 *  \brief Append a new element to the list, and return its list index.
 */
size_t simplelist_Append(simplelist_t *L, const void *element);

/**
 *  \brief Returns the length of the list.
 */
size_t simplelist_GetLength(const simplelist_t *L);

/**
 *  \brief Returns a given element from the list.
 */
const void *simplelist_GetElem(const simplelist_t *L, size_t i);

/**
 *  \brief Iterator object for a simple list.
 */
typedef struct tagSimplelistIterator_t
{
    const simplenode_t *cur_elem;
    size_t id;
} simplelist_iter_t;

/**
 *  \brief Initialize an iterator.
 */
void simplelist_InitIter(const simplelist_t *L, simplelist_iter_t *i);

/**
 *  \brief Initialize an iterator and return the first element.
 */
const void *simplelist_BeginIter(const simplelist_t *L,
                                 simplelist_iter_t *i);

/**
 *  \brief Return the object of the current iteration.
 */
const void *simplelist_GetIterObj(const simplelist_iter_t *i);

/**
 *  \brief Return the index of the object of the current iteration.
 */
size_t simplelist_GetIterId(const simplelist_iter_t *i);

/**
 *  \brief Advances the iterator, and then returns the new iteration
 *  object.
 */
const void *simplelist_NextIter(simplelist_iter_t *i);

/**
 *  \brief Return the last element in the list.
 */
const void *simplelist_GetLastElem(const simplelist_t *L);

