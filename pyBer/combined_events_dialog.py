"""Choose a named OR label without changing the original behavior choices."""
from PySide6 import QtCore, QtWidgets


class CombineEventsDialog(QtWidgets.QDialog):
    def __init__(self, names, selected=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine behaviors / zones")
        self.resize(520, 470)
        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel(
            "Combine as OR: active when any selected behavior or zone is active. "
            "Overlapping or touching bouts become one bout, without double counting. "
            "Original labels and source files are unchanged.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        search = QtWidgets.QLineEdit()
        search.setPlaceholderText("Filter behaviors / zones")
        layout.addWidget(search)
        from behavior_zone_dialog import BehaviorCheckList
        self.choices = BehaviorCheckList()
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if name in selected else QtCore.Qt.CheckState.Unchecked)
            self.choices.addItem(item)
        layout.addWidget(self.choices, 1)
        search.textChanged.connect(self._filter)
        layout.addWidget(QtWidgets.QLabel("Combined label name"))
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. Zone 1 or Zone 2")
        layout.addWidget(self.name)
        hint = QtWidgets.QLabel("Applied separately to loaded recordings containing every selected label. "
                               "Missing labels are not treated as zero. The result is available in PSTH and comparisons.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.create_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.create_button.setText("Create combined label")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.choices.itemChanged.connect(self._validate)
        self.name.textChanged.connect(self._validate)
        self._validate()

    def members(self):
        return [self.choices.item(i).text() for i in range(self.choices.count())
                if self.choices.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def _filter(self, value):
        for i in range(self.choices.count()):
            item = self.choices.item(i)
            item.setHidden(value.casefold() not in item.text().casefold())

    def _validate(self, *_args):
        self.create_button.setEnabled(len(self.members()) >= 2 and bool(self.name.text().strip()))
