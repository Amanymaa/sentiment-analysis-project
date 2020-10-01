def _import_all_modules():
    """dynamically imports all modules in the package"""
    import traceback
    import os
    global results
    globals_, locals_ = globals(), locals()

    def load_module(modulename, package_module):
        try:
            names = []
            module = __import__(package_module, globals_, locals_, [modulename])

            for name in module.__dict__:
                if not name.startswith('_'):
                    globals_[name] = module.__dict__[name]
                    names.append(name)
        except Exception:
            traceback.print_exc()
            raise

        return module, names

    def load_dir(abs_dirpath, rel_dirpath=''):
        results = []

        # dynamically import all the package modules
        for filename in os.listdir(abs_dirpath):
            rel_filepath = os.path.join(rel_dirpath, filename)
            abs_filepath = os.path.join(abs_dirpath, filename)

            if filename[0] != '_' and os.path.isfile(abs_filepath) and filename.split('.')[-1] in ('py', 'pyw'):
                modulename = '.'.join(os.path.normpath(os.path.splitext(rel_filepath)[0]).split(os.sep))
                package_module = '.'.join([__name__, modulename])

                module, names = load_module(modulename, package_module)
                results += names
            elif os.path.isdir(abs_filepath):
                results += load_dir(abs_filepath, rel_filepath)

        return results

    return load_dir(os.path.dirname(__file__))


__all__ = _import_all_modules()
del _import_all_modules
