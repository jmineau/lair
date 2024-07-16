.. currentmodule:: lair.uataq.filesystem

File System
===========

The UATAQ subpackage operates on the idea that instrument data is collected
differently by each research group, despite potentially being from the same
site or instrument. Furthermore, the format in which data is written, and
therefore read, is dependent on the system that logged the data rather than the
instrument itself. To address this, UATAQ introduces a ``filesystem`` module
which consists of :class:`GroupSpace` and :class:`DataFile` objects. The :class:`GroupSpace`
objects provide group-specific methods for working with data from that group,
while the :class:`DataFile` objects handle the actual parsing of the data files.

Group Spaces
------------

Each research group has its own module in ``groupspaces`` where group-specific code is stored.
Each group module must contain a subclass of :class:`GroupSpace` 
and define :class:`DataFile` subclasses  for each file format that the group uses.

All :class:`GroupSpace` objects are stored in the :data:`groups` dictionary with the group name as the key.

   The default research group can be changed at runtime via :data:`uataq.filesystem.DEFAULT_GROUP`
   or *permanently* (until the next update) changed in the the ``filesystem.__init__`` module:

   .. literalinclude:: /../../lair/uataq/filesystem/__init__.py
      :language: python
      :caption: lair/uataq/filesystem/__init__.py
      :lineno-start: 14
      :lines: 14


.. autosummary::
   :toctree: ../api
   :template: group.rst
   :recursive:

   ~groupspaces.horel
   ~groupspaces.lin

Contents
--------

- :data:`DEFAULT_GROUP`
- :data:`groups`
- :data:`lvls`
- :class:`DataFile`
- :class:`GroupSpace`
- :func:`filter_datafiles`
- :func:`parse_datafiles`

.. autodata:: DEFAULT_GROUP

.. autodata:: groups

.. autodata:: lvls

.. autoclass:: DataFile
   :members:

.. autoclass:: GroupSpace
   :members:
   :show-inheritance:

.. autofunction:: filter_datafiles

.. autofunction:: parse_datafiles
