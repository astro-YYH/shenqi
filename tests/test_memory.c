#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>

#include "stub.h"
#include "../memory.h"


static void
test_allocator(void ** state)
{
    Allocator A0[1];
    allocator_init(A0, "Default", 4096 * 1024, 1);

    void * p1 = allocator_alloc_bot(A0, "M+1", 1024);
    void * p2 = allocator_alloc_bot(A0, "M+2", 1024);

    void * q1 = allocator_alloc_top(A0, "M-1", 1024);
    void * q2 = allocator_alloc_top(A0, "M-2", 1024);

    allocator_print(A0);

    allocator_free(p2);
    allocator_free(q2);
    allocator_free(p1);
    allocator_free(q1);

    allocator_destroy(A0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_allocator),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
