/**
 *  @file test/placement.c
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Resolves positions in a device table built from the machine its arguments describe, and prints the
 *      placement as `<vendor> <generation>`, or `refused`.
 *
 *      fatwheels_test_placement cuda=80 ada=89;120 rocm=gfx942 cuda:sm_120 cuda:sm_90 rocm:gfx942 0 1
 *
 *  - `<generation>=<codes>` lists the codes the build compiled that generation for, as CMake reports them.
 *  - `<vendor>:<architecture>` adds a device of that vendor.
 *  - `<position>` is a device to place, by its position in the table.
 *
 *  The table lists the devices as @ref fw_devices would: NVIDIA's, then AMD's, leaving out a vendor whose floor has
 *  no codes, as a build without it does. A device loads a generation when its architecture runs one of the
 *  generation's codes, as the driver's loader decides.
 */
#include <stdio.h>  // `printf`, `fprintf`, `snprintf`
#include <stdlib.h> // `strtol`, `strtoul`
#include <string.h> // `strchr`, `strcmp`, `strncmp`, `strtok`

#include "fatwheels/fatwheels.h"

enum { cuda_k, rocm_k, vendors_k, devices_max_k = 64 };

static char const *const vendor_names[vendors_k] = {"cuda", "rocm"};

static struct {
    char const *name;
    fw_capability_t capability;
    int vendor;
    char const *codes;
} generations[] = {{"cuda", fw_cap_cuda_k, cuda_k, NULL},
                   {"ampere", fw_cap_ampere_k, cuda_k, NULL},
                   {"ada", fw_cap_ada_k, cuda_k, NULL},
                   {"rocm", fw_cap_rocm_k, rocm_k, NULL},
                   {"cdna3", fw_cap_cdna3_k, rocm_k, NULL}};
enum { generations_k = sizeof(generations) / sizeof(generations[0]) };

/** Each vendor's devices, by architecture like `sm_90` or `gfx942`. */
static char const *machine[vendors_k][devices_max_k];
static int machine_counts[vendors_k];

/** Whether a device of @p architecture runs @p code, spelled as CMake lists them: `89`, `90a-real`, `gfx942`. */
static int runs(char const *architecture, char const *code) {
    if (strncmp(architecture, "sm_", 3) != 0) return strcmp(architecture, code) == 0;
    char *suffix = NULL;
    long const device = strtol(architecture + 3, NULL, 10), compiled = strtol(code, &suffix, 10);
    if (*suffix == 'a') return device == compiled; // arch-specific codes skip forward compatibility
    // PTX compiles for any later architecture, while `-real` SASS runs only on later minors of its own major
    return device >= compiled && (strcmp(suffix, "-real") != 0 || device / 10 == compiled / 10);
}

static fw_capability_t loaded_on(int vendor, char const *architecture) {
    unsigned loaded = 0;
    for (int i = 0; i != generations_k; ++i) {
        if (generations[i].vendor != vendor || !generations[i].codes) continue;
        char codes[256];
        snprintf(codes, sizeof(codes), "%s", generations[i].codes);
        for (char *code = strtok(codes, ";"); code; code = strtok(NULL, ";"))
            if (runs(architecture, code)) loaded |= generations[i].capability;
    }
    return (fw_capability_t)loaded;
}

static int generation_named(char const *name) {
    int i = 0;
    while (i != generations_k && strcmp(generations[i].name, name) != 0) ++i;
    return i;
}

static char const *name_of(fw_capability_t capability) {
    int i = 0;
    while (generations[i].capability != capability) ++i;
    return generations[i].name;
}

int main(int argc, char **argv) {
    fw_size_t positions[devices_max_k], count = 0;
    for (int i = 1; i != argc; ++i) {
        char *value = NULL;
        int known = 1;
        if ((value = strchr(argv[i], '='))) {
            *value++ = '\0';
            int const generation = generation_named(argv[i]);
            if ((known = generation != generations_k)) generations[generation].codes = value;
        }
        else if ((value = strchr(argv[i], ':'))) {
            *value++ = '\0';
            int vendor = 0;
            while (vendor != vendors_k && strcmp(vendor_names[vendor], argv[i]) != 0) ++vendor;
            known = vendor != vendors_k && machine_counts[vendor] != devices_max_k;
            if (known) machine[vendor][machine_counts[vendor]++] = value;
        }
        else if ((known = count != devices_max_k)) positions[count++] = (fw_size_t)strtoul(argv[i], NULL, 10);
        if (!known) return fprintf(stderr, "unknown or excess argument '%s'\n", argv[i]), 2;
    }

    fw_device_t table[vendors_k * devices_max_k];
    fw_size_t table_count = 0;
    for (int vendor = 0; vendor != vendors_k; ++vendor) {
        int const floor = generation_named(vendor_names[vendor]);
        if (!generations[floor].codes) continue;
        for (int ordinal = 0; ordinal != machine_counts[vendor]; ++ordinal)
            table[table_count++] =
                (fw_device_t) {generations[floor].capability, ordinal, loaded_on(vendor, machine[vendor][ordinal])};
    }
    fw_placement_t placement;
    if (fw_placement_resolve(table, table_count, positions, count, &placement) != fw_success_k)
        return puts("refused") < 0;
    printf("%s %s\n", name_of(placement.vendor), name_of(placement.generation));
    return placement.table != table || placement.positions != positions || placement.count != count;
}
