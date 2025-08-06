EPOCH = 0


def clear():
    global EPOCH
    EPOCH += 1


class cache:

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.cache = f'_{self.name}_cache'
        self.epoch = f'_{self.name}_epoch'
        try:
            from functools import update_wrapper
            update_wrapper(self, func)
        except:
            pass

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if getattr(instance, self.epoch, None) != EPOCH:
            value = self.func(instance)
            setattr(instance, self.cache, value)
            setattr(instance, self.epoch, EPOCH)
        return getattr(instance, self.cache)


# FREEZE = True


# class cache:

#     def __init__(self, func):
#         self.func = func
#         self.name = f'_{func.__name__}_cache'
#         try:
#             self.__doc__ = func.__doc__
#             self.__name__ = func.__name__
#         except:
#             pass

#     def __get__(self, instance, owner=None):
#         if instance is None:
#             return self
#         if not FREEZE:
#             val = self.func(instance)
#             setattr(instance, self.name, val)
#             return val
#         else:
#             if not hasattr(instance, self.name):
#                 setattr(instance, self.name, self.func(instance))
#             return getattr(instance, self.name)
