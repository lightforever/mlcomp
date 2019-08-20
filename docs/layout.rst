Layout
======================================

.. toctree::
   :maxdepth: 2


Layout is a mechanism in MLComp to customize your reports.

MLComp already includes some layouts.

You can find them at http://localhost:4201/reports/layouts

The structure is the following:

::

    extend: base # an optional field

    # To choose the best epoch
    metric:
      name: str
      minimize: True/False

    items: list of metrics to log during training
    layout: list of layout components

**extend**

Name of an existing layout. Items and Layouts will be merged

**items**

    Each item is a dictionary with a required key *type*.

    By key each item is available in layout components ( key field )

    Possible types:
        - series
            - key: str [required]
                source of a metric

**layout**

    Customizes appearance in the UI.

    **type** is required to choose a component

    Possible components:
        - panel
            - title: str [required]
            - expanded: True/False
            - parent_cols: int
            - cols: int
            - row_height: int
            - items: list of other components
            - table: True/False
                Use a table structure instead of Grid List. In this case child items are not put on new rows, but child components get more vertical space.
        - blank
            - cols
            - rows
        - series
            - source: str [required]
            - multi: True/False
                Either plot many series on a single plot or divide them into individual plots.
            - group: train/valid
                if you want to see only specific group
            - rows: int
            - cols: int
        - table
            - source: list[str]
                list of metrics to compare
            - rows: int
            - cols: int
        - img
            - source: str [required]
            - rows: int
            - cols: int