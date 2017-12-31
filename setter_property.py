

class SetterProperty:
    """
    Setter only property.
    Usage:

    class MyClass:
        def __init__(self):
            self._x = None
            self._y = None

        @SetterProperty
        def x(self, val):
            self._x = val

        @SetterProperty
        def x_and_y(self, vals):
            self._x = vals[0]
            self._y = vals[1]


    if __name__ == '__main__':
        obj = MyClass()
        obj.x = 5  # sets  obj._x to 5
        print(obj._x, obj._y) # 5, None
        obj.x_and_y = 1, 2 # sets obj._x to 1 and obj._y to 2
        print(obj._x, obj._y) # 1, 2

    """

    def __init__(self, func):
        self.func = func

    def __set__(self, obj, *values):
        return self.func(obj, *values)
