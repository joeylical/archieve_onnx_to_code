class MemManager():
    instance = None
    def __init__(self, size):
        self._size = size

    def alloc(self, size):
        pass

    def free(self, addr):
        pass

MemMgr = MemManager()

__all__ = MemMgr
