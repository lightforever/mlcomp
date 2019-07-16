class PaginatorOptions:
    def __init__(self,
                 page_number: int,
                 page_size: int,
                 sort_column: str = None,
                 sort_descending: bool = None):
        self.sort_column = sort_column
        self.sort_descending = sort_descending
        self.page_number = page_number
        self.page_size = page_size

        assert (page_number is not None and page_size) \
               or (not page_number is None and not page_size), \
            'Specify both page_number and page_size'

        if not sort_column:
            self.sort_column = 'id'
            self.sort_descending = True


__all__ = ['PaginatorOptions']
