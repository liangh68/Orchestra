from torch import distributed as dist


def is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


def print_by_rank(rank: int = 0, *args):
    if dist.is_initialized() and dist.get_rank() == rank:
        print(f"Rank[{rank}]:", end=" ")
        print(*args)
    elif not dist.is_initialized() and rank == 0:
        print(f"Rank[0]:", end=" ")
        print(*args)


def print_with_rank(*args):
    if dist.is_initialized():
        print(f"Rank[{dist.get_rank()}]:", end=" ")
        print(*args)
    else:
        print(f"Rank[0]:", end=" ")
        print(*args)

