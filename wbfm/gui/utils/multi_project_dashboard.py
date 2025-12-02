import argparse
from ast import arg
from collections import defaultdict
from glob import glob
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm.auto import tqdm
import yaml
import re
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QComboBox, QLabel, QStatusBar,
    QFileDialog, QMessageBox, QDialog, QTextEdit, QHeaderView, QMenu,
    QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from snakemake import snakemake as snakemake_api
import io
from contextlib import redirect_stdout, redirect_stderr


@dataclass
class SnakemakeStats:
    """Statistics from snakemake --summary"""
    total_rules: int = 0
    completed_rules: int = 0
    pending_rules: int = 0
    failed_rules: int = 0
    running: bool = False
    error_message: Optional[str] = None

    completed_rule_names: set = field(default_factory=set)
    pending_rule_names: set = field(default_factory=set)
    failed_rule_names: set = field(default_factory=set)


def make_rule_tooltip(stats: SnakemakeStats) -> str:
    def fmt(rules):
        return ", ".join(sorted(rules)) if rules else "None"
    return (
        f"Completed rules:\n{fmt(stats.completed_rule_names)}\n\n"
        f"Pending rules:\n{fmt(stats.pending_rule_names)}\n\n"
        f"Failed rules:\n{fmt(stats.failed_rule_names)}"
    )


@dataclass
class Project:
    """Represents a Snakemake project"""
    path: Path
    name: str
    config: Dict
    target_rule: str
    stats: Optional[SnakemakeStats] = None
    
    @classmethod
    def load_from_path(cls, project_path: Path):
        """Load a project from a path containing project_config.yaml"""
        config_file = project_path / "project_config.yaml"
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            project_name = config.get('project_name', project_path.name)
            target_rule = config.get('target_rule', 'traces')
            
            project = cls(
                path=project_path,
                name=project_name,
                config=config,
                target_rule=target_rule,
                stats=None
            )
            
            # Check if snakemake file exists
            snakefile = project_path / "snakemake" / "pipeline.smk"
            if not snakefile.exists():
                print(f"Warning: Snakefile not found at {snakefile}")
            
            return project
            
        except Exception as e:
            print(f"Error loading project from {project_path}: {e}")
            return None
        
    @staticmethod
    def find_newest_slurm_log(log_dir: str) -> Optional[str]:
        """
        Find the newest slurm_*.out file in the given directory.
        Returns full path or None if no matching files exist.
        """
        pattern = os.path.join(log_dir, "slurm-*.out")
        files = glob(pattern)
        if not files:
            return None
        # Sort by modification time, newest first
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    def update_status(self):
        """Update the project status by querying Snakemake"""
        snakefile = self.path / "snakemake" / "pipeline.smk"
        
        if not snakefile.exists():
            self.stats = SnakemakeStats(error_message="Snakefile not found")
            return
        
        # Check if jobs are running first
        running = self._check_if_running(snakefile)
        
        # Get pipeline statistics
        logfile = self.find_newest_slurm_log(self.path / "snakemake")
        if logfile is None:
            stats = SnakemakeStats(error_message="Log file not found")
            # stats = self._get_snakemake_stats_via_dryrun(snakefile)
        else:
            stats = self._get_snakemake_stats(logfile)
        stats.running = running
        
        self.stats = stats

        # print("="*100)
        # print(f"Updated status for project {self.name} at {self.path}")
    
    def _check_if_running(self, snakefile: Path) -> bool:
        """Check if Snakemake jobs are currently running using the Python API."""
        # Look at the most recent file update; if it was within the last 5 minutes, assume running
        snakemake_folder = self.path / "snakemake"
        
        try:
            latest_mtime = max(f.stat().st_mtime for f in Path(snakemake_folder).iterdir() if f.is_file())
            if (datetime.now().timestamp() - latest_mtime) < 300:
                return True
        except ValueError:
            # No slurm files found
            return False

    @staticmethod
    def _get_snakemake_stats(log_path: str) -> SnakemakeStats:
        stats = SnakemakeStats()
        in_job_table = False
        counts = {}
        job_id2name = defaultdict(lambda : "Unknown Rule")
        job_id_count = 0
        
        submitted_jobs = set()
        finished_jobs = set()
        failed_jobs = set()

        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()

                # ---- Job stats table ----
                if line.startswith("Job stats:"):
                    in_job_table = True
                    continue
                if in_job_table:
                    if line.startswith("----------------"):
                        continue
                    if not line:
                        in_job_table = False
                        continue
                    # parse lines like: rule_name <spaces> count
                    m = re.match(r"(\S+)\s+(\d+)$", line)
                    if m:
                        rule, count = m.groups()
                        if rule.lower() != "total":
                            counts[rule] = int(count)
                        else:
                            stats.total_rules = int(count)
                        job_id2name[job_id_count] = rule
                        job_id_count += 1
                    continue

                # ---- Finished jobs ----
                if re.search(r"Finished job \d+\.", line):
                    m = re.search(r"Finished job (\d+)\.", line)
                    if m:
                        finished_jobs.add(job_id2name[int(m.group(1))])
                    continue

                # ---- Submitted jobs ----
                if re.search(r"Submitted job \d+\b", line):
                    m = re.search(r"Submitted job (\d+)\b", line)
                    if m:
                        submitted_jobs.add(job_id2name[int(m.group(1))])
                    continue

                # ---- Failed jobs ----
                if line.startswith("ERROR:snakemake.logging:Error in rule"):
                    # extract rule name
                    m = re.search(r"Error in rule (\S+):", line)
                    if m:
                        failed_jobs.add(m.group(1))
                    else:
                        failed_jobs.add("unknown")
                    continue

                # ---- All jobs were finished before this log file ----
                if line.startswith("Nothing to be done"):
                    finished_jobs.add("Nothing to be done")
                    stats.total_rules = 1  # In this case the table won't be printed, so we should set it manually
                    break  # No need to search more

        # Sometimes jobs failed but then were re-run and finished
        failed_jobs.difference_update(finished_jobs)
        submitted_jobs.difference_update(finished_jobs)

        stats.completed_rules = len(finished_jobs)
        stats.failed_rules = len(failed_jobs)
        stats.pending_rules = len(submitted_jobs)

        stats.completed_rule_names = finished_jobs
        stats.failed_rule_names = failed_jobs
        # stats.pending_rule_names = all_rules - finished_jobs - failed_jobs

        return stats#, counts

    def _get_snakemake_stats_via_dryrun(self, 
        snakefile: str,
    ) -> SnakemakeStats:
        """Run a dry-run and parse the output to estimate workflow progress."""
        stats = SnakemakeStats()

        counts_completed = self._run_snakemake_dry_run_and_parse(snakefile)
        counts_total = self._run_snakemake_dry_run_and_parse(snakefile, forceall=True)

        stats.total_rules = counts_total['total']
        stats.pending_rules = counts_completed.get('total', 0)
        stats.completed_rules = stats.total_rules - stats.pending_rules
        stats.failed_rules = 0

        # print("="*100)
        # print(counts_completed)
        # print(counts_total)

        stats.running = False  # since this is a dry-run, nothing is actually running

        return stats

    def _run_snakemake_dry_run_and_parse(self, snakefile: Path, forceall=False) -> dict:

        # Capture Snakemake’s stdout/stderr into a string buffer
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                success = snakemake_api(
                    snakefile=snakefile,
                    workdir=self.path / "snakemake",
                    targets=[self.target_rule],
                    dryrun=True,
                    quiet=True,
                    printshellcmds=False,
                    cores=1,
                    keepgoing=True,
                    forceall=forceall
                )

            output = stdout_buf.getvalue() + stderr_buf.getvalue()
            counts = self._parse_quiet_output(output)


        except Exception as e:
            print(f"Error invoking Snakemake for {self.path}: {e}")

        finally:
            stdout_buf.close()
            stderr_buf.close()

        return counts

    @staticmethod
    def _parse_quiet_output(output: str) -> SnakemakeStats:
        counts = {}

        # Skip header lines and parse the table
        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("Job stats:") or line.startswith("---"):
                continue

            # Match lines like: rule_name <whitespace> count
            m = re.match(r"(\S+)\s+(\d+)$", line)
            if m:
                rule, count = m.groups()
                counts[rule] = int(count)
        return counts

    def _get_stats_from_list(self, snakefile: Path) -> SnakemakeStats:
        """Get statistics using --list and --list-input-changes"""
        stats = SnakemakeStats()
        
        try:
            # Get list of rules to be executed
            result = subprocess.run(
                ['snakemake', '--snakefile', str(snakefile), 
                 '--dry-run', '--quiet', '--printshellcmds', self.target_rule],
                cwd=str(self.path),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if "Nothing to be done" in result.stderr or "Nothing to be done" in result.stdout:
                stats.completed_rules = 1
                stats.total_rules = 1
                return stats
            
            # Count job mentions in dry-run output
            output = result.stdout + result.stderr
            
            # Parse "X jobs" from output
            job_match = re.search(r'(\d+)\s+job', output)
            if job_match:
                pending = int(job_match.group(1))
                stats.pending_rules = pending
                stats.total_rules = pending
            
            return stats
            
        except Exception as e:
            stats.error_message = str(e)[:200]
            return stats
    
    def get_completion_percentage(self) -> int:
        """Calculate overall completion percentage"""
        if not self.stats or self.stats.total_rules == 0:
            return 0
        
        return int((self.stats.completed_rules / self.stats.total_rules) * 100)
    
    def get_overall_status(self) -> str:
        """Get overall project status"""
        if not self.stats:
            return "unknown"
        
        if self.stats.error_message:
            return "error"
        
        if self.stats.running:
            return "running"
        
        if self.stats.failed_rules > 0:
            return "crashed"
        
        if self.stats.pending_rules > 0:
            return "pending"
        
        if self.stats.completed_rules == self.stats.total_rules and self.stats.total_rules > 0:
            return "completed"
        
        return "unknown"
    
    def get_last_modified(self) -> datetime:
        """Get last modification time of project directory"""
        try:
            return datetime.fromtimestamp(self.path.stat().st_mtime)
        except:
            return datetime.now()
    
    def get_status_details(self) -> str:
        """Get detailed status string"""
        if not self.stats:
            return "Not checked yet"
        
        if self.stats.error_message:
            return f"Error: {self.stats.error_message}"
        
        return (f"Total: {self.stats.total_rules} | "
                f"Completed: {self.stats.completed_rules} | "
                f"Pending: {self.stats.pending_rules} | "
                f"Failed: {self.stats.failed_rules}")


class ProjectScanner:
    """Scans directories for projects"""
    
    @staticmethod
    def scan_directory(root_path: Path) -> List[Project]:
        """Recursively scan directory for projects"""
        projects = []
        
        try:
            for config_file in root_path.rglob("project_config.yaml"):
                project_path = config_file.parent
                project = Project.load_from_path(project_path)
                
                if project:
                    projects.append(project)
        
        except Exception as e:
            print(f"Error scanning directory {root_path}: {e}")
        
        return projects


class StatusUpdateThread(QThread):
    """Thread for updating project status without blocking GUI (supports parallelism)"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal()
    
    def __init__(self, projects: List[Project]):
        super().__init__()
        self.projects = projects

    def run(self):
        total = len(self.projects)
        # Use ThreadPoolExecutor for parallel status updates
        def update_and_emit(idx_proj):
            idx, project = idx_proj
            project.update_status()
            self.progress.emit(idx + 1, total)

        for idx, project in tqdm(enumerate(self.projects), total=total, desc="Updating project statuses"):
            update_and_emit((idx, project))

        self.finished.emit()


class ProjectDetailsDialog(QDialog):
    """Dialog to show detailed project information"""
    
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle(f"Project Details: {self.project.name}")
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        # Project info
        info_label = QLabel(
            f"<b>Project:</b> {self.project.name}<br>"
            f"<b>Path:</b> {self.project.path}<br>"
            f"<b>Target Rule:</b> {self.project.target_rule}<br>"
            f"<b>Snakefile:</b> {self.project.path / 'snakemake' / 'pipeline.smk'}"
        )
        layout.addWidget(info_label)
        
        # Status details
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        
        content = "Pipeline Status:\n" + "="*60 + "\n\n"
        
        if self.project.stats:
            stats = self.project.stats
            
            if stats.running:
                content += "🔄 Status: RUNNING\n\n"
            elif stats.error_message:
                content += f"⚠️ Status: ERROR\n\nError Message:\n{stats.error_message}\n\n"
            else:
                status = self.project.get_overall_status()
                status_symbols = {
                    "completed": "✓ COMPLETED",
                    "pending": "○ PENDING",
                    "crashed": "⚠️ FAILED",
                    "unknown": "? UNKNOWN"
                }
                content += f"{status_symbols.get(status, status.upper())}\n\n"
            
            content += f"Total Rules: {stats.total_rules}\n"
            content += f"Completed: {stats.completed_rules}\n"
            content += f"Pending: {stats.pending_rules}\n"
            content += f"Failed: {stats.failed_rules}\n"
            content += f"Completion: {self.project.get_completion_percentage()}%\n\n"
            
            # Show snakemake command examples
            content += "\nUseful Commands:\n" + "-"*60 + "\n"
            content += f"# Run pipeline:\n"
            content += f"cd {self.project.path}\n"
            content += f"snakemake --snakefile snakemake/pipeline.smk {self.project.target_rule}\n\n"
            content += f"# Dry run:\n"
            content += f"snakemake --snakefile snakemake/pipeline.smk --dry-run {self.project.target_rule}\n\n"
            content += f"# Show summary:\n"
            content += f"snakemake --snakefile snakemake/pipeline.smk --summary {self.project.target_rule}\n"
        else:
            content += "Status not yet checked. Click 'Refresh' to update.\n"
        
        details_text.setPlainText(content)
        layout.addWidget(details_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.clicked.connect(self.refresh_status)
        button_layout.addWidget(refresh_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def refresh_status(self):
        """Refresh the project status"""
        self.project.update_status()
        self.setup_ui()  # Rebuild UI with new data


class ProjectStatusGUI(QMainWindow):
    """Main GUI window for project status monitoring"""

    def __init__(self, target_folder: Optional[str] = None, auto_refresh=True):
        super().__init__()
        self.projects: List[Project] = []
        self.root_path: Optional[Path] = Path(target_folder) if target_folder else None
        self.current_view = "tree"  # tree or table
        self.filter_status = "all"
        self.update_thread: Optional[StatusUpdateThread] = None
        
        self.setup_ui()
        if auto_refresh:
            self.setup_timer()

        if self.root_path is not None:
            self.update_path_label(self.root_path)
    
    def setup_ui(self):
        self.setWindowTitle("Snakemake Project Status Monitor")
        self.setGeometry(100, 100, 1400, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        # Folder selection
        self.path_label = QLabel("No folder selected")
        self.path_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; }")
        toolbar_layout.addWidget(self.path_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        toolbar_layout.addWidget(browse_btn)
        
        toolbar_layout.addStretch()
        
        # Default target rule
        target_label = QLabel("Default Target:")
        toolbar_layout.addWidget(target_label)
        
        self.target_input = QLineEdit()
        self.target_input.setText("traces")
        self.target_input.setPlaceholderText("e.g., traces, all")
        self.target_input.setMaximumWidth(150)
        self.target_input.returnPressed.connect(self.update_default_target)
        toolbar_layout.addWidget(self.target_input)
        
        toolbar_layout.addStretch()
        
        # View mode buttons
        self.tree_view_btn = QPushButton("Tree View")
        self.tree_view_btn.setCheckable(True)
        self.tree_view_btn.setChecked(True)
        self.tree_view_btn.clicked.connect(lambda: self.switch_view("tree"))
        toolbar_layout.addWidget(self.tree_view_btn)
        
        self.table_view_btn = QPushButton("Table View")
        self.table_view_btn.setCheckable(True)
        self.table_view_btn.clicked.connect(lambda: self.switch_view("table"))
        toolbar_layout.addWidget(self.table_view_btn)
        
        toolbar_layout.addStretch()
        
        # Filter
        filter_label = QLabel("Filter:")
        toolbar_layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Completed", "Running", "Crashed", "Pending"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        toolbar_layout.addWidget(self.filter_combo)
        
        # Search
        search_label = QLabel("Search:")
        toolbar_layout.addWidget(search_label)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search projects...")
        self.search_box.textChanged.connect(self.apply_search)
        toolbar_layout.addWidget(self.search_box)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh All")
        self.refresh_btn.clicked.connect(self.refresh_projects)
        toolbar_layout.addWidget(self.refresh_btn)
        
        main_layout.addLayout(toolbar_layout)
        
        # Progress label (hidden by default)
        self.progress_label = QLabel()
        self.progress_label.hide()
        main_layout.addWidget(self.progress_label)
        
        # Tree View
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Projects")
        self.tree_widget.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self.show_tree_context_menu)
        main_layout.addWidget(self.tree_widget)
        
        # Table View
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels(
            ["Project Name", "Path", "Target", "Progress", "Status", "Details"]
        )
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_widget.setSortingEnabled(True)
        self.table_widget.itemDoubleClicked.connect(self.on_table_item_double_clicked)
        self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.show_table_context_menu)
        self.table_widget.hide()
        main_layout.addWidget(self.table_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status_bar()
    
    def setup_timer(self):
        """Setup auto-refresh timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_projects)
        self.timer.start(5*60000)  # Refresh every 5 minutes
    
    def browse_folder(self):
        """Open folder browser dialog"""
        folder = QFileDialog.getExistingDirectory(self, "Select Root Folder")
        
        if folder:
            self.root_path = Path(folder)
            self.update_path_label(self.root_path)

    def update_path_label(self, path: Path):
        """Update the displayed path label"""
        self.path_label.setText(str(path))
        self.scan_projects()
    
    def update_default_target(self):
        """Update default target rule for new projects"""
        # This will be used when scanning for new projects
        pass
    
    def scan_projects(self):
        """Scan for projects in root path"""
        if not self.root_path:
            return
        
        self.projects = ProjectScanner.scan_directory(self.root_path)
        
        # Update target rules from config or use default
        default_target = self.target_input.text() or "traces"
        for project in self.projects:
            if 'target_rule' not in project.config:
                project.target_rule = default_target
        
        self.update_displays()
        self.refresh_projects()
    
    def refresh_projects(self):
        """Refresh project status in background thread"""
        if not self.projects or self.update_thread and self.update_thread.isRunning():
            return
        
        self.refresh_btn.setEnabled(False)
        self.progress_label.show()
        self.progress_label.setText("Updating project status...")
        
        self.update_thread = StatusUpdateThread(self.projects)
        self.update_thread.progress.connect(self.on_update_progress)
        self.update_thread.finished.connect(self.on_update_finished)
        self.update_thread.start()
    
    def on_update_progress(self, current: int, total: int):
        """Update progress during status refresh"""
        self.progress_label.setText(f"Updating project status... ({current}/{total})")
    
    def on_update_finished(self):
        """Handle completion of status update"""
        self.progress_label.hide()
        self.refresh_btn.setEnabled(True)
        self.update_displays()
        self.update_status_bar()
    
    def switch_view(self, view_mode: str):
        """Switch between tree and table view"""
        self.current_view = view_mode
        
        if view_mode == "tree":
            self.tree_view_btn.setChecked(True)
            self.table_view_btn.setChecked(False)
            self.tree_widget.show()
            self.table_widget.hide()
        else:
            self.tree_view_btn.setChecked(False)
            self.table_view_btn.setChecked(True)
            self.tree_widget.hide()
            self.table_widget.show()
    
    def update_displays(self):
        """Update both tree and table displays"""
        self.update_tree_view()
        self.update_table_view()
    
    def update_tree_view(self):
        """Update tree view with current projects"""
        self.tree_widget.clear()
        
        if not self.root_path:
            return
        
        # Build folder structure
        folder_items = {}
        
        for project in self.projects:
            if not self.should_show_project(project):
                continue
            
            # Get relative path
            rel_path = project.path.relative_to(self.root_path)
            path_parts = rel_path.parts
            
            # Create folder hierarchy
            current_parent = self.tree_widget.invisibleRootItem()
            current_path = self.root_path
            
            for part in path_parts[:-1]:  # All but the project folder
                current_path = current_path / part
                path_key = str(current_path)
                
                if path_key not in folder_items:
                    folder_item = QTreeWidgetItem(current_parent, [f"📁 {part}"])
                    folder_item.setData(0, Qt.UserRole, {"type": "folder", "path": current_path})
                    folder_items[path_key] = folder_item
                    current_parent = folder_item
                else:
                    current_parent = folder_items[path_key]
            
            # Add project item
            status = project.get_overall_status()
            progress = project.get_completion_percentage()
            
            status_icons = {
                "completed": "✓",
                "running": "⟳",
                "pending": "○",
                "crashed": "⚠️",
                "error": "❌",
                "unknown": "?"
            }
            
            icon = status_icons.get(status, "?")
            dots = "●" * (progress // 20) + "○" * (5 - progress // 20)
            
            project_item = QTreeWidgetItem(
                current_parent,
                [f"📊 {project.name} [{dots}] {progress}% {icon}"]
            )
            project_item.setData(0, Qt.UserRole, {"type": "project", "project": project})
            
            # Color coding
            if status == "completed":
                project_item.setForeground(0, QColor(0, 150, 0))
            elif status == "crashed" or status == "error":
                project_item.setForeground(0, QColor(200, 0, 0))
            elif status == "running":
                project_item.setForeground(0, QColor(0, 100, 200))
            
            # Tooltip with detailed stats
            if project.stats:
                project_item.setToolTip(0, make_rule_tooltip(project.stats))
        
        self.tree_widget.expandAll()
    
    def update_table_view(self):
        """Update table view with current projects"""
        self.table_widget.setRowCount(0)
        self.table_widget.setSortingEnabled(False)
        
        for project in self.projects:
            if not self.should_show_project(project):
                continue
            
            row = self.table_widget.rowCount()
            self.table_widget.insertRow(row)
            
            # Project name
            name_item = QTableWidgetItem(project.name)
            name_item.setData(Qt.UserRole, project)
            self.table_widget.setItem(row, 0, name_item)
            
            # Path
            path_item = QTableWidgetItem(str(project.path))
            self.table_widget.setItem(row, 1, path_item)
            
            # Target rule
            target_item = QTableWidgetItem(project.target_rule)
            self.table_widget.setItem(row, 2, target_item)
            
            # Progress
            progress = project.get_completion_percentage()
            progress_item = QTableWidgetItem(f"{progress}%")
            progress_item.setData(Qt.UserRole, progress)
            self.table_widget.setItem(row, 3, progress_item)
            
            # Status
            status = project.get_overall_status()
            status_icons = {
                "completed": "✓ Done",
                "running": "⟳ Running",
                "pending": "○ Pending",
                "crashed": "⚠️ Failed",
                "error": "❌ Error",
                "unknown": "? Unknown"
            }
            status_text = status_icons.get(status, "Unknown")
            status_item = QTableWidgetItem(status_text)
            self.table_widget.setItem(row, 4, status_item)
            
            # Color coding
            if status == "completed":
                status_item.setForeground(QColor(0, 150, 0))
            elif status == "crashed" or status == "error":
                status_item.setForeground(QColor(200, 0, 0))
            elif status == "running":
                status_item.setForeground(QColor(0, 100, 200))
            
            # Details
            details = project.get_status_details()
            details_item = QTableWidgetItem(details)
            self.table_widget.setItem(row, 5, details_item)

            # Tooltip with detailed stats
            if project.stats:
                name_item.setToolTip(make_rule_tooltip(project.stats))
        
        self.table_widget.setSortingEnabled(True)
    
    def should_show_project(self, project: Project) -> bool:
        """Check if project matches current filter and search"""
        # Apply filter
        status = project.get_overall_status()
        
        if self.filter_status == "completed" and status != "completed":
            return False
        elif self.filter_status == "running" and status != "running":
            return False
        elif self.filter_status == "crashed" and status not in ["crashed", "error"]:
            return False
        elif self.filter_status == "pending" and status != "pending":
            return False
        
        # Apply search
        search_text = self.search_box.text().lower()
        if search_text:
            if search_text not in project.name.lower() and search_text not in str(project.path).lower():
                return False
        
        return True
    
    def apply_filter(self, filter_text: str):
        """Apply status filter"""
        self.filter_status = filter_text.lower()
        self.update_displays()
        self.update_status_bar()
    
    def apply_search(self, search_text: str):
        """Apply search filter"""
        self.update_displays()
    
    def update_status_bar(self):
        """Update status bar with project statistics"""
        total = len(self.projects)
        
        if total == 0:
            self.status_bar.showMessage("No projects found")
            return
        
        completed = sum(1 for p in self.projects if p.get_overall_status() == "completed")
        running = sum(1 for p in self.projects if p.get_overall_status() == "running")
        crashed = sum(1 for p in self.projects if p.get_overall_status() in ["crashed", "error"])
        pending = sum(1 for p in self.projects if p.get_overall_status() == "pending")
        
        self.status_bar.showMessage(
            f"{total} projects | "
            f"✓ {completed} completed | "
            f"⟳ {running} running | "
            f"⚠️ {crashed} crashed | "
            f"○ {pending} pending"
        )
    
    def on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on tree item"""
        data = item.data(0, Qt.UserRole)
        
        if not data:
            return
        
        if data["type"] == "project":
            self.open_project_folder(data["project"])
        elif data["type"] == "folder":
            # Toggle expand/collapse
            item.setExpanded(not item.isExpanded())
    
    def on_table_item_double_clicked(self, item: QTableWidgetItem):
        """Handle double-click on table item"""
        row = item.row()
        project = self.projects[row]
        self.open_project_folder(project)
    
    def open_project_folder(self, project: Project):
        """Open project folder in file explorer"""
        path = os.path.abspath(str(project.path))
        
        if sys.platform == "win32":
            subprocess.Popen(["explorer", path], shell=True)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    
    def show_tree_context_menu(self, position):
        """Show context menu for tree view"""
        item = self.tree_widget.itemAt(position)
        
        if not item:
            return
        
        data = item.data(0, Qt.UserRole)
        
        if not data:
            return
        
        menu = QMenu()
        
        if data["type"] == "project":
            project = data["project"]
            open_action = menu.addAction("Open Project Folder")
            details_action = menu.addAction("View Details")
            
            action = menu.exec_(self.tree_widget.viewport().mapToGlobal(position))
            
            if action == open_action:
                self.open_project_folder(project)
            elif action == details_action:
                self.show_project_details(project)
        elif data["type"] == "folder":
            folder_path = data["path"]
            open_action = menu.addAction("Open Folder")
            
            action = menu.exec_(self.tree_widget.viewport().mapToGlobal(position))
            
            if action == open_action:
                self.open_folder(folder_path)
    
    def show_table_context_menu(self, position):
        """Show context menu for table view"""
        item = self.table_widget.itemAt(position)
        
        if not item:
            return
        
        row = item.row()
        project = self.projects[row]
        
        menu = QMenu()
        
        open_action = menu.addAction("Open Project Folder")
        details_action = menu.addAction("View Details")
        
        action = menu.exec_(self.table_widget.viewport().mapToGlobal(position))
        
        if action == open_action:
            self.open_project_folder(project)
        elif action == details_action:
            self.show_project_details(project)
    
    def open_folder(self, folder_path: Path):
        """Open folder in file explorer"""
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder_path])
        else:
            subprocess.run(["xdg-open", folder_path])
    
    def show_project_details(self, project: Project):
        """Show project details in a dialog"""
        details = f"Name: {project.name}\nPath: {project.path}\nStatus: {project.status}"
        QMessageBox.information(self, "Project Details", details)


def main(target_folder: str):
    app = QApplication(sys.argv)
    gui = ProjectStatusGUI(target_folder)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Snakemake Project Status Monitor")
    args.add_argument("--folder", type=str, help="Root folder to scan for projects")
    parsed_args = args.parse_args()
    target_folder = parsed_args.folder

    main(target_folder)