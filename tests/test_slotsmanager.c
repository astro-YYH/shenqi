#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include "allvars.h"
#include "domain.h"
#include "slotsmanager.h"
#include "mymalloc.h"
#include "stub.h"
#include "allvars.h"

struct particle_data *P;
struct global_data_all_processes All;
int NTask, ThisTask;
int NumPart;

static int
setup_particles(void ** state)
{
    All.MaxPart = 1024;
    NumPart = 128 * 6;

    int newSlots[6] = {128, 128, 128, 128, 128, 128};

    P = (struct particle_data *) mymalloc("P", All.MaxPart * sizeof(struct particle_data));
    memset(P, 0, sizeof(struct particle_data) * All.MaxPart);

    slots_init();
    int ptype;
    slots_set_enabled(0, sizeof(struct sph_particle_data));
    slots_set_enabled(4, sizeof(struct star_particle_data));
    slots_set_enabled(5, sizeof(struct bh_particle_data));
    for(ptype = 1; ptype < 4; ptype++) {
        slots_set_enabled(ptype, sizeof(struct particle_data_ext));
    }

    slots_reserve(1, newSlots);

    int i;
    for(i = 0; i < NumPart; i ++) {
        P[i].ID = i;
        P[i].Type = i / (NumPart / 6);
    }
    slots_setup_topology();

    slots_setup_id();

    return 0;
}

static int
teardown_particles(void **state)
{
    slots_free();
    myfree(P);
    return 0;
}

static void
test_slots_gc(void **state)
{
    setup_particles(state);
    int i;
    int compact[6];
    for(i = 0; i < 6; i ++) {
        slots_mark_garbage(128 * i);
        compact[i] = 1;
    }
    slots_gc(compact);
    assert_int_equal(NumPart, 127 * i);

    assert_int_equal(SlotsManager->info[0].size, 127);
    assert_int_equal(SlotsManager->info[4].size, 127);
    assert_int_equal(SlotsManager->info[5].size, 127);

    teardown_particles(state);
    return;
}

static void
test_slots_gc_sorted(void **state)
{
    setup_particles(state);
    int i;
    for(i = 0; i < 6; i ++) {
        slots_mark_garbage(128 * i);
    }
    slots_gc_sorted();
    assert_int_equal(NumPart, 127 * i);

    assert_int_equal(SlotsManager->info[0].size, 127);
    assert_int_equal(SlotsManager->info[4].size, 127);
    assert_int_equal(SlotsManager->info[5].size, 127);

    teardown_particles(state);
    return;
}

static void
test_slots_reserve(void **state)
{
    /* FIXME: these depends on the magic numbers in slots_reserve. After
     * moving those numbers to All.* we shall rework the code here. */
    setup_particles(state);

    int newSlots[6] = {128, 128, 128, 128, 128, 128};
    int oldSize[6];
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        oldSize[ptype] = SlotsManager->info[ptype].maxsize;
    }
    slots_reserve(1, newSlots);

    /* shall not increase max size*/
    for(ptype = 0; ptype < 6; ptype++) {
        assert_int_equal(oldSize[ptype], SlotsManager->info[ptype].maxsize);
    }

    for(ptype = 0; ptype < 6; ptype++) {
        newSlots[ptype] += 1;
    }

    /* shall not increase max size; because it is small difference */
    slots_reserve(1, newSlots);
    for(ptype = 0; ptype < 6; ptype++) {
        assert_int_equal(oldSize[ptype], SlotsManager->info[ptype].maxsize);
    }

    for(ptype = 0; ptype < 6; ptype++) {
        newSlots[ptype] += 128;
    }

    /* shall increase max size; because it large difference */
    slots_reserve(1, newSlots);

    for(ptype = 0; ptype < 6; ptype++) {
        assert_true(oldSize[ptype] < SlotsManager->info[ptype].maxsize);
    }

}

/*Check that we behave correctly when the slot is empty*/
static void
test_slots_zero(void **state)
{
    setup_particles(state);
    int i;
    int compact[6] = {1,0,0,0,1,1};
    for(i = 0; i < NumPart; i ++) {
        slots_mark_garbage(i);
    }
    slots_gc(compact);
    assert_int_equal(NumPart, 0);
    assert_int_equal(SlotsManager->info[0].size, 0);
    assert_int_equal(SlotsManager->info[1].size, 128);
    assert_int_equal(SlotsManager->info[4].size, 0);
    assert_int_equal(SlotsManager->info[5].size, 0);

    teardown_particles(state);

    setup_particles(state);
    for(i = 0; i < NumPart; i ++) {
        slots_mark_garbage(i);
    }
    slots_gc_sorted();
    assert_int_equal(NumPart, 0);
    assert_int_equal(SlotsManager->info[0].size, 0);
    assert_int_equal(SlotsManager->info[4].size, 0);
    assert_int_equal(SlotsManager->info[5].size, 0);

    teardown_particles(state);

    return;

}

static void
test_slots_fork(void **state)
{
    setup_particles(state);
    int i;
    for(i = 0; i < 6; i ++) {
        slots_fork(128 * i, P[i * 128].Type);
    }

    assert_int_equal(NumPart, 129 * i);

    assert_int_equal(SlotsManager->info[0].size, 129);
    assert_int_equal(SlotsManager->info[4].size, 129);
    assert_int_equal(SlotsManager->info[5].size, 129);

    teardown_particles(state);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_slots_gc),
        cmocka_unit_test(test_slots_gc_sorted),
        cmocka_unit_test(test_slots_reserve),
        cmocka_unit_test(test_slots_fork),
        cmocka_unit_test(test_slots_zero),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
