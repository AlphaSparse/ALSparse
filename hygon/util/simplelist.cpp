/**
 *   \brief Simple read-only, lockable, append-only list implementation.
 */

#include <assert.h>
#include <stdio.h>

#include "alphasparse/util/simplelist.h"
#include "alphasparse/util/malloc.h"

#if defined(ALPHA_ENABLE_MUTEX)
#error "*** \\todo: Enable pthreads support! ***"
#else
/** Prevent other processes from modifying L. Reading is still permitted. */
#define simplelist_Lock(L)

/** Release L so that other processes may modify it. */
#define simplelist_Unlock(L)
#endif

/**
 *  \brief
 *
 *  \returns A newly allocated simple list.
 */
simplelist_t *
simplelist_Create(void)
{
    simplelist_t *new_list = alpha_malloc(sizeof(simplelist_t) * 1);
    if (new_list != NULL)
    {
        new_list->head = NULL;
        new_list->tail = NULL;
    }
    return new_list;
}

/**
 *  \brief
 *
 *  \param[in,out] L List object to destroy.
 *
 *  In multithreaded environments, the caller MUST ensure that
 *  no other threads are executing any operations on L.
 */
void simplelist_Destroy(simplelist_t *L)
{
    if (L != NULL)
    {
        simplenode_t *cur_node = L->head;
        while (cur_node != NULL)
        {
            simplenode_t *next_node = cur_node->next;
            alpha_free(cur_node);
            cur_node = next_node;
        }
    }
    alpha_free(L);
}

/**
 *  \brief
 *
 *  \param[in,out] L List object.
 *  \param[in] element A pointer to a read-only element to insert.
 *  \returns The index i >= 1 of the new list element. That is,
 *  element becomes the logical list element, L[i]. On error,
 *  returns 0.
 *
 *  In multithreaded environments, calls to #simplelist_Append are
 *  mutually exclusive. Simultaneous #simplelist_GetLength and
 *  #simplelist_GetElem operations may execute and are considered
 *  'safe' provided they access elements whose index is less than
 *  the length of the list at the time of the call to
 *  #simplelist_Append.
 */
size_t
simplelist_Append(simplelist_t *L, const void *element)
{
    simplenode_t *new_node;
    size_t new_index;

    assert(L != NULL);

    /* Create a new node. */
    new_node = alpha_malloc(sizeof(simplenode_t) * 1);
    if (new_node == NULL)
        return 0;

    new_node->element = element;
    new_node->next = NULL;

    /** Begin mutually exclusive region. */
    // simplelist_Lock(L);

    if (L->num_elements == 0) /* empty list */
    {
        L->head = new_node;
    }
    else
    {
        assert(L->tail != NULL);
        L->tail->next = new_node;
    }
    L->tail = new_node;
    L->num_elements++;
    new_index = L->num_elements;

    // simplelist_Unlock(L);
    /** End mutually exclusive region. */

    return new_index;
}

/**
 *  \brief
 *
 *  \param[in] L List object.
 *  \returns Length of the list, or 0 if the list is either
 *  empty or NULL.
 */
size_t
simplelist_GetLength(const simplelist_t *L)
{
    return (L == NULL) ? 0 : L->num_elements;
}

/**
 *  \brief
 *
 *  \param[in] L List object containing \f$n\f$ elements.
 *  \param[in] i Index of item to retrieve, starting at 1 for
 *  the first element.
 *
 *  If L is invalid or i \f$> n\f$, then NULL is
 *  returned. Otherwise, returns the logical element L[i].
 */
const void *
simplelist_GetElem(const simplelist_t *L, size_t i)
{
    size_t length;
    size_t k;
    const simplenode_t *cur_node;

    if ((length = simplelist_GetLength(L)) == 0)
        return NULL;
    if (i <= 0 || i > length)
        return NULL;

    for (k = 1, cur_node = L->head; k < i; k++, cur_node = cur_node->next)
        ; /* cur_node == L[k] */

    return cur_node->element;
}

/**
 *  \brief
 *
 *  \param[in] L List over which to iterate.
 *  \param[in,out] i Pointer to an uninitialized iteration object.
 */
void simplelist_InitIter(const simplelist_t *L, simplelist_iter_t *i)
{
    if (L == NULL || i == NULL)
        return;
    i->cur_elem = L->head;
    if (i->cur_elem != NULL)
        i->id = 1;
}

const void *
simplelist_BeginIter(const simplelist_t *L, simplelist_iter_t *i)
{
    simplelist_InitIter(L, i);
    return simplelist_GetIterObj(i);
}

/**
 *  \brief
 *
 *  \param[in] i Pointer to an initialized iteration object.
 */
const void *
simplelist_GetIterObj(const simplelist_iter_t *i)
{
    return (i == NULL || i->cur_elem == NULL) ? NULL : i->cur_elem->element;
}

/**
 *  \brief
 *
 *  \param[in] i Pointer to an initialized iteration object.
 */
size_t
simplelist_GetIterId(const simplelist_iter_t *i)
{
    return (i == NULL) ? 0 : i->id;
}

/**
 *  \brief
 *
 *  \param[in,out] i Pointer to an initialized iteration object.
 */
const void *
simplelist_NextIter(simplelist_iter_t *i)
{
    if (i != NULL && i->cur_elem != NULL)
    {
        i->cur_elem = i->cur_elem->next;
        i->id = (i->cur_elem == NULL) ? 0 : (i->id + 1);
    }
    return simplelist_GetIterObj(i);
}

const void *
simplelist_GetLastElem(const simplelist_t *L)
{
    if (L == NULL || L->tail == NULL)
        return NULL;
    else
        return L->tail->element;
}

