import argparse
import sys
from pathlib import Path

from PyQt5 import QtWidgets
from ruamel.yaml import YAML

from wbfm.utils.external.utils_yaml import load_config


MISSING = "<missing>"


def parse_key_path(key_path):
    key_path = key_path.strip().replace(".", "/")
    return [k for k in key_path.split("/") if k]


def get_nested(cfg, keys):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def set_nested(cfg, keys, value):
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_value(raw):
    s = raw.strip()
    if s in ("", "null", "None", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def scan_parent_folder(parent_folder, yaml_name, keys):
    rows = []
    parent = Path(parent_folder)
    for sub in sorted(parent.iterdir()):
        if not sub.is_dir():
            continue
        fname = sub / yaml_name
        if not fname.exists():
            continue
        try:
            cfg = load_config(str(fname))
        except Exception as e:
            rows.append((sub.name, f"<error: {e}>", str(fname), False))
            continue
        val, found = get_nested(cfg, keys)
        display = MISSING if not found or val is None else str(val)
        rows.append((sub.name, display, str(fname), found))
    return rows


def apply_value_to_files(rows, keys, value):
    yaml = YAML()
    updated = []
    for _, _, fname, _ in rows:
        with open(fname, "r") as f:
            cfg = yaml.load(f)
        if cfg is None:
            cfg = {}
        set_nested(cfg, keys, value)
        with open(fname, "w") as f:
            yaml.dump(cfg, f)
        updated.append(fname)
    return updated


class BulkEditConfigGui(QtWidgets.QMainWindow):
    def __init__(self, parent_folder="", yaml_name="project_config.yaml", key_path="physical_units/exposure_time"):
        super().__init__()
        self.setWindowTitle("Bulk edit config parameter")
        self.resize(700, 500)
        self.rows = []

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        folder_row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit(parent_folder)
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        folder_row.addWidget(QtWidgets.QLabel("Parent folder:"))
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        form = QtWidgets.QFormLayout()
        self.yaml_edit = QtWidgets.QLineEdit(yaml_name)
        self.key_edit = QtWidgets.QLineEdit(key_path)
        form.addRow("Yaml file:", self.yaml_edit)
        form.addRow("Key path:", self.key_edit)
        layout.addLayout(form)

        scan_btn = QtWidgets.QPushButton("Scan")
        scan_btn.clicked.connect(self.scan)
        layout.addWidget(scan_btn)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Project", "Current value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        new_row = QtWidgets.QHBoxLayout()
        self.new_value_edit = QtWidgets.QLineEdit()
        self.new_value_edit.setPlaceholderText("New value, e.g. 12")
        apply_btn = QtWidgets.QPushButton("Set all to new value")
        apply_btn.clicked.connect(self.preview_new_value)
        new_row.addWidget(QtWidgets.QLabel("New value:"))
        new_row.addWidget(self.new_value_edit)
        new_row.addWidget(apply_btn)
        layout.addLayout(new_row)

        save_btn = QtWidgets.QPushButton("Save to all files")
        save_btn.clicked.connect(self.save)
        layout.addWidget(save_btn)

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

    def browse_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select parent folder", self.folder_edit.text())
        if d:
            self.folder_edit.setText(d)

    def scan(self):
        parent = self.folder_edit.text().strip()
        yaml_name = self.yaml_edit.text().strip() or "project_config.yaml"
        keys = parse_key_path(self.key_edit.text())
        if not parent or not Path(parent).is_dir():
            self.status.setText("Pick a valid parent folder")
            return
        if not keys:
            self.status.setText("Enter a key path, e.g. physical_units/exposure_time")
            return
        self.rows = scan_parent_folder(parent, yaml_name, keys)
        self.table.setRowCount(len(self.rows))
        for i, (name, display, _, _) in enumerate(self.rows):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(display))
        self.status.setText(f"Found {len(self.rows)} projects with {yaml_name}")

    def preview_new_value(self):
        raw = self.new_value_edit.text()
        if not raw.strip():
            self.status.setText("Enter a new value first")
            return
        for i in range(self.table.rowCount()):
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(raw.strip()))
        self.status.setText("Preview updated; press Save to write to disk")

    def save(self):
        if not self.rows:
            self.status.setText("Scan first")
            return
        keys = parse_key_path(self.key_edit.text())
        table_values = [self.table.item(i, 1).text() if self.table.item(i, 1) else "" for i in range(self.table.rowCount())]
        yaml = YAML()
        n = 0
        for (_, _, fname, _), raw in zip(self.rows, table_values):
            if raw == MISSING or raw.startswith("<error"):
                continue
            with open(fname, "r") as f:
                cfg = yaml.load(f)
            if cfg is None:
                cfg = {}
            set_nested(cfg, keys, parse_value(raw))
            with open(fname, "w") as f:
                yaml.dump(cfg, f)
            n += 1
        self.status.setText(f"Saved {n} files")
        self.scan()


def main():
    parser = argparse.ArgumentParser(description="Bulk view and edit one yaml parameter across projects in a folder")
    parser.add_argument("--parent_folder", "-p", default="")
    parser.add_argument("--yaml_name", default="project_config.yaml")
    parser.add_argument("--key_path", default="physical_units/exposure_time")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = BulkEditConfigGui(args.parent_folder, args.yaml_name, args.key_path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
