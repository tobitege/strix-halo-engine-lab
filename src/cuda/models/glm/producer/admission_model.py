"""CPU-only capacity estimate. No model reads, CUDA imports, or device discovery.

Workspace arithmetic mirrors the bounded native layouts; confirm its estimate
against residency.workspace_bytes in the next authorized execution receipt.
Host availability is a post-load observation, not total RAM minus model size.
"""
import argparse
import json

GIB = 1 << 30


def aligned(n):
    return (n+255)//256*256


def moe(rows, experts, hidden, inter, c1=True):
    n = rows*(8 if experts > 1 else 1)
    q = min(16, experts)*hidden*inter*2
    sizes = [experts*4, (experts+1)*4, experts*4, n*4, 17*4, 4]
    sizes += [q]*3+[n*hidden*2]*2+[n*inter*2, n*inter*4, n*hidden*4]
    if c1:
        sizes += [q]*3+[n*max(hidden, inter)*4, n*inter*4, 32768]
    return sum(map(aligned, sizes))


def dsa(cap):
    r = 64
    sizes = [64, 64, r*4096*4]
    sizes += [n*4 for n in (4096*1536, 1536*4096, 128*4096, 128*4096, 32*4096)]
    sizes += [i*o*2 for i, o in ((4096, 1536), (1536, 16384), (4096, 512), (16384, 4096), (1536, 4096))]
    sizes += [r*16384*2, r*16384*4]
    sizes += [r*n*4 for n in (1536, 1536, 512, 128, 128, 32, 4096, 16384, 64*512, 64*512, 16384)]
    sizes += [r*(cap//4)*4]*2+[r*64*512*2, r*9*64*512*4, r*9*64*2*4, r*64*2051*4, r*2051*4, r*4]
    return sum(map(aligned, sizes))


def kda(rows):
    c, h, heads, b, dim = 8192, 4096, 64, 32, 128
    sizes = [64, h*3*c*2, rows*max(h, c)*2, rows*max(h, 3*c)*4, rows*3*c*4]
    sizes += [rows*c*4]*2+[rows*heads*4]+[rows*dim*4]*2+[rows*heads*4*dim*4]+[rows*c*4]*2
    sizes += [heads*b*dim*4]+[heads*b*b*4]*2+[heads*b*dim*4]*3
    return sum(map(aligned, sizes))


def workspace(rows, cap):
    m = max(moe(rows, 288, 4096, 2048, c1) for c1 in (False, True))
    mtp = sum(map(aligned, [64, 8192*4096*2, rows*8192*2, rows*4096*4, rows*8192*4]
                               + [rows*4096*4]*6+[rows*288*4, rows*8*4, rows*8*4, dsa(cap), m]))
    return max(m, moe(rows, 1, 4096, 12288), kda(rows), dsa(cap), mtp)


def activation(cap, tile):
    return cap*4096*4*5+tile*4096*4*14+tile*24*4+(cap+1)*4+tile*64+4


def estimate(cap, available, commit, floor, file_gib, tile):
    device = 20 if cap <= 8192 else 19
    remaining = 43-device-13
    budget = max(0, min(available, commit)-floor)
    extra = min(remaining, int(budget//file_gib))
    return dict(capacity=cap, row_tile=tile, device_layers_assumed=device,
                pinned_layers=13+extra, file_layers=remaining-extra,
                extra_pinned_gib=extra*file_gib, floor_gib=floor,
                post_load_available_gib=available, post_load_commit_gib=commit,
                workspace_bytes=workspace(tile, cap), activation_bytes=activation(cap, tile),
                qualification='ESTIMATE; actual allocation and Windows commit decide admission')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--total-gib', type=float, default=127.8)
    p.add_argument('--available-after-13-gib', type=float, default=32e9/GIB)
    p.add_argument('--commit-after-13-gib', type=float, default=32e9/GIB)
    p.add_argument('--layer-gib', type=float, default=37.2/11)
    a = p.parse_args()
    print(json.dumps(dict(total_host_gib=a.total_gib,
                         assumptions='32 decimal GB free after 13 pinned layers; mean file layer 37.2/11 GiB; device counts held fixed; rings already charged',
                         cases=[estimate(cap, a.available_after_13_gib, a.commit_after_13_gib, floor, a.layer_gib, tile)
                                for cap in (8192, 32768) for floor in (16, 8) for tile in (256, 2048)]), indent=2))
