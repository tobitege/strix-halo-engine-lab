// kfdevmon: LD_PRELOAD probe for KFD queue evictions of the process it is loaded into (no root needed).
// - reads the process's own KFD SMI events (queue eviction with trigger, restore, unmap, VM fault)
//   through ROCr's /dev/kfd fd;
// - logs every userptr / GTT / VRAM KFD allocation and free (ioctl interposition) with a backtrace
//   for userptr ones and the /proc/self/maps line that owns the user address.
// Env: KFDEVMON_OUT=<prefix> (log = <prefix>.<pid>), KFDEVMON_BT=0 disables backtraces.
// Build: gcc -O2 -shared -fPIC -o kfdevmon.so kfdevmon.c -ldl -lpthread
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

static FILE* out;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int (*real_ioctl)(int, unsigned long, ...);
static int want_bt = 1;
#define MAXBO 65536
static struct { uint64_t handle, size, uaddr; uint32_t flags; } bos[MAXBO];
static int nbo;
static uint64_t up_alive_n, up_alive_bytes, up_total_n, gtt_alive_bytes;

static double now_boot(void) { struct timespec t; clock_gettime(CLOCK_BOOTTIME, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void maps_line(uint64_t a, char* buf, size_t n) {
    FILE* m = fopen("/proc/self/maps", "r"); buf[0] = 0; if (!m) return;
    char l[512];
    while (fgets(l, sizeof l, m)) {
        unsigned long lo, hi; if (sscanf(l, "%lx-%lx", &lo, &hi) != 2) continue;
        if (a >= lo && a < hi) { l[strcspn(l, "\n")] = 0; snprintf(buf, n, "%s", l); break; }
    }
    fclose(m);
}

static void log_alloc(struct kfd_ioctl_alloc_memory_of_gpu_args* a, int rc) {
    if (rc != 0) return;
    const int up = (a->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0;
    pthread_mutex_lock(&mu);
    if (nbo < MAXBO) { bos[nbo].handle = a->handle; bos[nbo].size = a->size; bos[nbo].uaddr = up ? a->mmap_offset : 0; bos[nbo].flags = a->flags; nbo++; }
    if (up) { up_alive_n++; up_alive_bytes += a->size; up_total_n++; }
    else if (a->flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) gtt_alive_bytes += a->size;
    if (up || a->size >= (64u << 20)) {
        char ml[400] = ""; if (up) maps_line(a->mmap_offset, ml, sizeof ml);
        fprintf(out, "%.6f ALLOC %s size=%llu flags=%x uaddr=%llx handle=%llx alive_up=%llu/%lluMB | %s\n", now_boot(),
                up ? "USERPTR" : (a->flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) ? "VRAM" : (a->flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) ? "GTT" : "OTHER",
                (unsigned long long)a->size, a->flags, (unsigned long long)(up ? a->mmap_offset : 0), (unsigned long long)a->handle,
                (unsigned long long)up_alive_n, (unsigned long long)(up_alive_bytes >> 20), ml);
        if (up && want_bt) {
            void* f[24]; int k = backtrace(f, 24); char** s = backtrace_symbols(f, k);
            for (int i = 2; s && i < k; ++i) fprintf(out, "    bt %s\n", s[i]);
            free(s);
        }
        fflush(out);
    }
    pthread_mutex_unlock(&mu);
}

static void log_free(uint64_t handle) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < nbo; ++i) if (bos[i].handle == handle) {
        const int up = (bos[i].flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0;
        if (up) { up_alive_n--; up_alive_bytes -= bos[i].size; }
        else if (bos[i].flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) gtt_alive_bytes -= bos[i].size;
        if (up || bos[i].size >= (64u << 20)) {
            fprintf(out, "%.6f FREE %s size=%llu uaddr=%llx alive_up=%llu/%lluMB\n", now_boot(), up ? "USERPTR" : "BO",
                    (unsigned long long)bos[i].size, (unsigned long long)bos[i].uaddr, (unsigned long long)up_alive_n, (unsigned long long)(up_alive_bytes >> 20));
            fflush(out);
        }
        bos[i] = bos[--nbo]; break;
    }
    pthread_mutex_unlock(&mu);
}

int ioctl(int fd, unsigned long req, ...) {
    va_list ap; va_start(ap, req); void* arg = va_arg(ap, void*); va_end(ap);
    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    uint64_t free_handle = 0;
    if (out && req == AMDKFD_IOC_FREE_MEMORY_OF_GPU) free_handle = ((struct kfd_ioctl_free_memory_of_gpu_args*)arg)->handle;
    const int rc = real_ioctl(fd, req, arg);
    if (out && req == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) log_alloc((struct kfd_ioctl_alloc_memory_of_gpu_args*)arg, rc);
    if (out && free_handle && rc == 0) log_free(free_handle);
    return rc;
}

static int find_kfd_fd(void) {
    DIR* d = opendir("/proc/self/fd"); if (!d) return -1; struct dirent* e; int r = -1; char p[64], t[256];
    while ((e = readdir(d))) {
        snprintf(p, sizeof p, "/proc/self/fd/%s", e->d_name); ssize_t n = readlink(p, t, sizeof t - 1);
        if (n > 0) { t[n] = 0; if (!strcmp(t, "/dev/kfd")) { r = atoi(e->d_name); break; } }
    }
    closedir(d); return r;
}

static uint32_t gpu_id(void) {
    for (int i = 0; i < 16; ++i) {
        char p[96]; snprintf(p, sizeof p, "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", i);
        FILE* f = fopen(p, "r"); if (!f) continue; unsigned v = 0; if (fscanf(f, "%u", &v) != 1) v = 0; fclose(f);
        if (v) return v;
    }
    return 0;
}

static void* smi_thread(void* unused) {
    (void)unused; int kfd = -1;
    for (int i = 0; i < 12000 && kfd < 0; ++i) { kfd = find_kfd_fd(); if (kfd < 0) usleep(50000); }
    if (kfd < 0) { fprintf(out, "# no /dev/kfd fd\n"); fflush(out); return NULL; }
    usleep(200000);
    struct kfd_ioctl_smi_events_args a = {.gpuid = gpu_id(), .anon_fd = 0};
    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    if (real_ioctl(kfd, AMDKFD_IOC_SMI_EVENTS, &a) != 0) { fprintf(out, "# SMI_EVENTS failed errno=%d\n", errno); fflush(out); return NULL; }
    uint64_t mask = KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_VMFAULT) | KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_START) |
        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_END) | KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_PAGE_FAULT_START) |
        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_QUEUE_EVICTION) | KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_QUEUE_RESTORE) |
        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_UNMAP_FROM_GPU);
    if (write(a.anon_fd, &mask, sizeof mask) != sizeof mask) { fprintf(out, "# mask write failed errno=%d\n", errno); fflush(out); return NULL; }
    fprintf(out, "# smi on gpuid=%u kfd_fd=%d boot=%.6f\n", a.gpuid, kfd, now_boot()); fflush(out);
    char buf[8192];
    for (;;) {
        struct pollfd p = {.fd = a.anon_fd, .events = POLLIN}; if (poll(&p, 1, -1) <= 0) continue;
        ssize_t n = read(a.anon_fd, buf, sizeof buf - 1); if (n <= 0) continue; buf[n] = 0;
        pthread_mutex_lock(&mu);
        for (char* l = strtok(buf, "\n"); l; l = strtok(NULL, "\n"))
            fprintf(out, "%.6f SMI %s | alive_up=%llu/%lluMB gtt=%lluMB\n", now_boot(), l, (unsigned long long)up_alive_n,
                    (unsigned long long)(up_alive_bytes >> 20), (unsigned long long)(gtt_alive_bytes >> 20));
        fflush(out); pthread_mutex_unlock(&mu);
    }
    return NULL;
}

__attribute__((constructor)) static void init(void) {
    const char* o = getenv("KFDEVMON_OUT"); if (!o || !*o) return;
    const char* bt = getenv("KFDEVMON_BT"); want_bt = !(bt && bt[0] == '0');
    char p[512]; snprintf(p, sizeof p, "%s.%d", o, (int)getpid()); out = fopen(p, "w"); if (!out) return;
    struct timespec r; clock_gettime(CLOCK_REALTIME, &r);
    fprintf(out, "# kfdevmon pid=%d boot=%.6f real=%ld.%09ld\n", (int)getpid(), now_boot(), (long)r.tv_sec, r.tv_nsec); fflush(out);
    pthread_t t; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_create(&t, &at, smi_thread, NULL);
}
