class BaseViewModel:
    def __init__(self):
        self._is_loading = False

    @property
    def is_loading(self) -> bool:
        return self._is_loading

    @is_loading.setter
    def is_loading(self, value: bool):
        self._is_loading = value

    def to_dict(self):
        return {"is_loading": self.is_loading}
