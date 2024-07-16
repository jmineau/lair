Configuration
=============

The UATAQ configuration is defined as a ``json`` file which specifies UATAQ
site characteristics including name, location, status, research groups
collecting data, and installed instruments.

.. warning::

    Changes to UATAQ site infrastructure, including instrument
    installation/removal, must be reflected in the configuration
    file for the lab to be able to properly access the data.

.. literalinclude:: /../../lair/uataq/config.json
    :language: json
    :caption: lair/uataq/config.json

