
class PyLCError(BaseException):
    pass


class PyLCLibraryError(PyLCError):
    pass


class PyLCFileError(PyLCError):
    pass


class PyLCProcessError(PyLCError):
    pass


class PyLCCancelled(PyLCProcessError):
    """Raised when a host application requests cooperative cancellation."""

    pass


class PyLCInputError(PyLCError):
    pass
