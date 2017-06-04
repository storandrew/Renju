#! /usr/bin/env python3

import lib

class ErrorHandler(object):
    def __enter__(self):
        pass

    def __exit__(self, error_type, error_args, traceback):
        if error_type is None:
            return
        if error_type is lib.gui.tk.TclError:
            print("WARNING: A GUI-related `TclError` error occurred:"
                    , error_args)
            return True
        if issubclass(error_type, Exception):
            lib.gui.Message(message=repr(error_args)
                    , icon='warning'
                    , title='Gomoku - Error').show()
        return False

if __name__ == '__main__':
    main_window = lib.gui.MainWindow(15, 15)
    with ErrorHandler():
        main_window.mainloop()
