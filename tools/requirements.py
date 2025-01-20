"""A script to generate the minimal requirements of a python package.

See the docstring of the tyro_cli function for more information.
"""

import importlib
from importlib.metadata import distribution, version
import inspect
import os
import pkgutil
import sys
import traceback
from typing import Dict, Generator, List, Mapping, Optional, Tuple
from types import ModuleType

import isort


def package(obj):
    """Return the name of the package in which an object is defined."""
    mod = inspect.getmodule(obj)
    if mod is None:
        return None
    base, _sep, _stem = mod.__name__.partition('.')
    return sys.modules[base].__name__


def imports(m: ModuleType, /) -> Generator[str, None, None]:
    """Get the packages imported by a module.
    
    Yields the names of the packages in which each member
    of the module is was defined in, excluding the given
    module itself. NOTE: this may yield duplicates, so the
    output is usually passed to set.
    """
    if not inspect.ismodule(m):
        raise TypeError(f'Expected a module, got {type(m)}')
    # Do not filter for modules since we want to also include packages
    # needed by from ... import ... statements.
    for name, member in inspect.getmembers(m):
        pkg = package(member)
        # print(f'Inspecting member {name}: {member} in package {pkg}')
        if pkg is not None and pkg != m.__name__:
            yield pkg

# def imports_recursive(m: ModuleType, /, _exclude: Optional[set] = None) -> Generator[ModuleType, None, None]:
#     if _exclude is None:
#         _exclude = set()
#     if not inspect.ismodule(m):
#         return
#     for name, member in inspect.getmembers(m):
#         if name in _exclude:
#             continue
#         if hasattr(member, '__path__'):
#             print(f'Inspecting package {name}: {member.__path__}')
#         elif hasattr(member, '__file__'):
#             print(f'Inspecting module {name}: {member.__file__}')
#         n = None
#         if inspect.ismodule(member):
#             n = name
#         # Detect 'from ... import ...' statements
#         elif hasattr(member, '__module__') and member.__module__ != m.__name__:
#             n = member.__module__
#         if n is not None:
#             _exclude.add(n)
#             yield n
#             yield from imports_recursive(member, _exclude)

# def pip_freeze(include_editable=False) -> Dict[str, str]:
#     req_lines = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split('\n')
#     reqs = {}
#     for line in req_lines:
#         line = line.strip()
#         if line == '':
#             continue
#         elif '-e ' in line:
#             if include_editable:
#                 reqs[line.split('-e ')[1]] = 'editable'
#         elif '==' in line:
#             name, ver = line.split('==')
#             reqs[name] = ver
#         else:
#             raise NotImplementedError(f'Cannot parse line from the output of pip freeze: {repr(line)}')
#     return reqs

def dependencies(module: str, /, ignore: Optional[set] = None) -> Dict[str, str]:
    """Find the dependencies of a module.
    
    The module is given as a string to be passed to importlib.import_module.

    Dependencies are identified as all installed distribution packages that
    provide one or more modules imported at the top level of the given module.
    Installed distributions are found using
    importlib.metadata.packages_distributions.
    """
    # print(f'[dependencies] Importing module {module}')
    m = importlib.import_module(module)
    imported = set(imports(m))
    # print(f'Imported packages: {imported}')
    if ignore is None:
        ignore = set()

    # Ignore builtin modules and the package of the module.
    ignore |= set(sys.builtin_module_names)
    pkg = package(m)
    if pkg is not None:
        # print(f'Ignoring source package {pkg}')
        ignore.add(pkg)

    # print(f'Builtin modules: {ignore}')
    pkg_dists = packages_distributions()
    requirements = {}
    for name in imported:
        # Replace this with sys.stdlib_module_names when upgrading to Python 3.10
        if name in ignore or isort.place_module(name) == 'STDLIB':
            continue
        # print(f'Inspecting package {name}')
        try:
            reqs = pkg_dists[name]
        except KeyError as e:
            # print(f'Error: {name}')
            # If we don't find the package, try to import it. If successful,
            # this means this a local package and we should ignore it.
            try:
                importlib.import_module(name)
            except ImportError:
                # If we still don't find the package, try to use the import name
                # as the distribution name.
                try:
                    dist = distribution(name)
                except Exception as e:
                    raise RuntimeError(f'Import package {name} is not provided by any known distribution package') from e
                print(f'Warning: Import package {name} is provided by distribution package with entry points {dist.entry_points}', file=sys.stderr)
                reqs = [name]
            else:
                print(f'Warning: Skipping local package {name}', file=sys.stderr)
                continue
        for req in reqs:
            ver = version(req)
            assert ver == requirements.setdefault(req, ver)
    return requirements


# TODO consider using modulefinder?
def modules(path: str) -> Generator[str, None, None]:
    def onerror(name):
        print(f"Error importing module {name}, traceback:")
        print(traceback.format_exc())
    # This is necessary since walk_packages tries to import modules
    # using their names relative to the package. The path is added
    # at the front to make sure subpackages of the package have precedence
    # over other installed packages with the same name.
    sys.path.insert(0, path)
    for info in pkgutil.walk_packages([path], onerror=onerror):
        yield info.name
    assert path == sys.path.pop(0)


def requirements(path: str, /) -> Dict[str, str]:
    """Return the requirements of an (import) package.
    
    Requirements are found by importing each module in the package
    and inspecting its top-level members: if a member is a module
    that is also found among installed packages (as reported by
    importlib.metadata.packages_distributions()), then the
    distribution package(s) that provide that module are
    considered required, and they are included in the output.
    """
    name = os.path.basename(path if not path.endswith('/') else path[:-1])
    # print(f'Ignoring source package {name}')
    cwd = os.getcwd()
    remove_cwd = False
    if cwd not in sys.path:
        remove_cwd = True
        sys.path.append(cwd)
    requirements = {}
    for m in modules(path):
        # print(f'[requirements] Inspecting module {m}')
        requirements.update(dependencies(m, ignore={name}))
    if remove_cwd:
        sys.path.remove(cwd)
    return requirements


def tyro_cli(
    path: str,
    /,
    include: Tuple[str, ...] = (),
    exclude: Tuple[str, ...] = (),
    toml: bool = False,
    include_python: bool = False
):
    """Print the requirements of a package.

    This is meant to be a better version of 'pip freeze' to
    generate the requirements for a package, as it only
    includes distribution packages that are explicitly imported
    by modules in the package.
    
    By deafult, the requirements are printed in the
    'requirements.txt' format. Other formats can be selected by
    setting the appropriate flag.

    It may be useful to pipe the output of this command to
    another command that copies it to the clipboard, e.g.

    $ python requirements.py some/package --toml | xsel -ib

    IMPORTANT: this will import all the (sub)packages contained
    in the package, since it calls pkgutil.walk_packages.

    Args:
        path: The path to the package. This is passed directly
            to pkgutil.walk_packages.
        include: A tuple of strings that specify the *DISTRIBUTION* names of
            packages that must be always included in the output, even if they
            are not explicitely imported anywhere in the package.
        toml: If True, print the requirements in the TOML format
            as a list named 'dependencies'. This is usually
            pasted inside a 'pyproject.toml' file.
        include_python: If True, include the python version (major and minor)
            as a comment in the output, e.g. '# python=3.8'.
    """
    if include_python:
        print(f'# python={sys.version_info.major}.{sys.version_info.minor}')
    if toml:
        print('dependencies = [')
    reqs = requirements(path)
    for pkg in include:
        if pkg not in reqs:
            reqs[pkg] = version(pkg)
    for pkg in exclude:
        reqs.pop(pkg, None)
    for req, ver in reqs.items():
        if toml:
            print(f'  "{req}=={ver}",')
        else:
            print(f'{req}=={ver}')
    if toml:
        print(']')

def clize_cli(path: str, /, include_python: bool = False):
    tyro_cli(path, include_python=include_python)


def main():
    try:
        import tyro
    except ImportError:
        try:
            import clize
        except ImportError:
            raise RuntimeError(
                'Either tyro or clize must be installed to run this command.'
                'You can install the preferred version with either `pip '
                'install towls.reqs[tyro]` or `pip install towls.reqs[clize]`'
            ) from None
        clize.run(clize_cli)
    tyro.cli(tyro_cli)


# ---------------------------------------------------------------------------
# Standalone copy of Python 3.10+'s importlib.metadata.packages_distributions


import collections
from importlib.metadata import distributions
from typing import List, Mapping


def packages_distributions() -> Mapping[str, List[str]]:
    """
    Return a mapping of top-level packages to their
    distributions.

    >>> import collections.abc
    >>> pkgs = packages_distributions()
    >>> all(isinstance(dist, collections.abc.Sequence) for dist in pkgs.values())
    True

    From https://github.com/python/cpython/blob/3.11/Lib/importlib/metadata/__init__.py#L1063
    """
    pkg_to_dist = collections.defaultdict(list)
    for dist in distributions():
        for pkg in _top_level_declared(dist) or _top_level_inferred(dist):
            pkg_to_dist[pkg].append(dist.metadata['Name'])
    return dict(pkg_to_dist)

def _top_level_declared(dist):
    return (dist.read_text('top_level.txt') or '').split()


def _top_level_inferred(dist):
    return {
        f.parts[0] if len(f.parts) > 1 else f.with_suffix('').name
        for f in always_iterable(dist.files)
        if f.suffix == ".py"
    }

def always_iterable(obj, base_type=(str, bytes)):
    """always_iterable from CPython 3.10's ._itertools

    If *obj* is iterable, return an iterator over its items::

        >>> obj = (1, 2, 3)
        >>> list(always_iterable(obj))
        [1, 2, 3]

    If *obj* is not iterable, return a one-item iterable containing *obj*::

        >>> obj = 1
        >>> list(always_iterable(obj))
        [1]

    If *obj* is ``None``, return an empty iterable:

        >>> obj = None
        >>> list(always_iterable(None))
        []

    By default, binary and text strings are not considered iterable::

        >>> obj = 'foo'
        >>> list(always_iterable(obj))
        ['foo']

    If *base_type* is set, objects for which ``isinstance(obj, base_type)``
    returns ``True`` won't be considered iterable.

        >>> obj = {'a': 1}
        >>> list(always_iterable(obj))  # Iterate over the dict's keys
        ['a']
        >>> list(always_iterable(obj, base_type=dict))  # Treat dicts as a unit
        [{'a': 1}]

    Set *base_type* to ``None`` to avoid any special handling and treat objects
    Python considers iterable as iterable:

        >>> obj = 'foo'
        >>> list(always_iterable(obj, base_type=None))
        ['f', 'o', 'o']
    
    From https://github.com/python/cpython/blob/3.11/Lib/importlib/metadata/_itertools.py#L23
    """
    if obj is None:
        return iter(())

    if (base_type is not None) and isinstance(obj, base_type):
        return iter((obj,))

    try:
        return iter(obj)
    except TypeError:
        return iter((obj,))


# ---------------------------------------------------------------------------
# execution as script

if __name__ == '__main__':
    main()

