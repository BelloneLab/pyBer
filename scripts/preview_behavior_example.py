"""Open an isolated pyBer example with explicitly simulated fiber signals.

python scripts/preview_behavior_example.py --mamir /path/behavior_frames.csv
Without --mamir, both behavior events and fiber signals are simulated.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import tempfile


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamir", type=Path, help="Read real MAMIR behavior times; fiber remains simulated")
    parser.add_argument("--zone-workbook", type=Path,
                        help="Also load EthoVision arena sheets for zone testing; fiber remains simulated")
    parser.add_argument("--smoke", action="store_true", help="Compute and check the example, then exit")
    parser.add_argument("--screenshot", type=Path, help="Save the example window to a PNG")
    parser.add_argument("--width", type=int, default=1900, help="Example window width")
    parser.add_argument("--height", type=int, default=1120, help="Example window height")
    parser.add_argument("--group", action="store_true", help="Open the existing Group view instead of Individual")
    args = parser.parse_args()
    if args.smoke:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["PYBER_SMOKE_TEST"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

    with tempfile.TemporaryDirectory(prefix="pyber-behavior-example-") as temporary:
        for variable, folder in (("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"),
                                 ("XDG_DATA_HOME", "data")):
            os.environ[variable] = str(Path(temporary) / folder)

        import numpy as np
        from PySide6 import QtCore, QtWidgets
        from analysis_core import ProcessedTrial
        from mamir_import import inspect_mamir, load_mamir
        import main

        settings_type = QtCore.QSettings

        class ExampleSettings(settings_type):
            def __init__(self, *_args, **_kwargs):
                super().__init__(str(Path(temporary) / "settings.ini"), settings_type.Format.IniFormat)

        QtCore.QSettings = ExampleSettings

        class ExamplePanel(main.PostProcessingPanel):
            def _restore_project_autosave_if_needed(self):
                pass

            def _autosave_project_cache_path(self):
                return str(Path(temporary) / "example-autosave.h5")

        main.PostProcessingPanel = ExamplePanel

        class ExampleWindow(main.MainWindow):
            def _maybe_show_first_run_tutorial(self):
                pass

        app = QtWidgets.QApplication([])
        main.apply_app_palette(app, "dark")
        window = ExampleWindow()
        window.resize(args.width, args.height)
        window.move(100, 70)
        panel = window.post_tab
        window.tabs.setCurrentWidget(panel)

        if args.mamir:
            identities = inspect_mamir(args.mamir, "behavior")["identities"][:2]
            sources = [load_mamir(args.mamir, "behavior", identity) for identity in identities]
            provenance = f"Real MAMIR behavior times from {args.mamir.name}; fiber signals are simulated."
        else:
            events = {
                "fighting": {"on": np.array([10., 30., 50.]), "off": np.array([13., 34., 53.])},
                "grooming": {"on": np.array([20., 40., 65.]), "off": np.array([24., 44., 69.])},
            }
            for event in events.values():
                event["dur"] = event["off"] - event["on"]
            sources = [copy.deepcopy({"kind": "binary_columns", "time": np.array([]),
                                      "behaviors": {}, "event_behaviors": events, "trajectory": {},
                                      "import_report": {"format": "demo", "identity": str(i + 1)}})
                       for i in range(2)]
            provenance = "Simulated behavior times and simulated fiber signals."

        if not sources:
            raise ValueError("No animal identities found in the supplied MAMIR file.")
        nonempty = [{name for name, event in source["event_behaviors"].items()
                     if len(event["on"])} for source in sources]
        common = set.intersection(*nonempty)
        available_labels = common or set.union(*nonempty)
        preferred = ["fighting", "grooming", "walking", "running", "rearing", "stretch_attend"]
        demo_labels = list(dict.fromkeys(name for name in preferred + sorted(available_labels)
                                        if name in available_labels))[:2]
        if not demo_labels:
            raise ValueError("This export contains no active behavior bouts for the example.")
        for index, info in enumerate(sources):
            events = info["event_behaviors"]
            end = max((float(np.max(event["off"])) for event in events.values()
                       if len(event["off"])), default=80.) + 5.
            time = np.arange(0., end, .05)
            signal = np.random.default_rng(5200 + index).normal(0., .04, len(time))
            signal += .04 * np.sin(time * .7)
            # Deliberately constructed responses illustrate the analysis only.
            for label, amplitude in zip(demo_labels, (.8, -.4)):
                for start, stop in zip(events.get(label, {}).get("on", []),
                                       events.get(label, {}).get("off", [])):
                    signal[(time >= start) & (time < stop)] += amplitude * (1. + .15 * index)
            path = f"DEMO_SIMULATED_FIBER_{index + 1}.csv"
            panel._processed.append(ProcessedTrial(
                path=path, channel_id="DEMO", time=time, raw_signal=signal.copy(),
                raw_reference=np.zeros_like(signal), output=signal, output_label="Simulated units"))
            info.setdefault("import_report", {}).update(paired_index=index, paired_path=path)
            panel._behavior_sources[Path(path).stem] = info

        if args.zone_workbook:
            arenas = panel._inspect_arena_workbook(str(args.zone_workbook))
            if not arenas:
                raise ValueError("No arena sheets found in the zone workbook.")
            for index in range(len(sources)):
                arena = arenas[index % len(arenas)]
                panel._add_generic_behavior_zone_file(
                    str(args.zone_workbook), "zone", sheet_name=arena["sheet"],
                    arena_options=arenas, target_index=index)
            provenance += " EthoVision zones are loaded from a separate experiment for UI testing only."

        panel.tab_sources.setCurrentIndex(1)
        panel._refresh_behavior_list()
        panel._set_event_category("behavior")
        panel.tab_visual_mode.setCurrentIndex(1 if args.group else 0)
        available = {panel.combo_behavior_name.itemText(i) for i in range(panel.combo_behavior_name.count())}
        selected = demo_labels[0]
        panel.combo_behavior_name.setCurrentText(selected)
        panel.combo_behavior_name.activated.emit(panel.combo_behavior_name.currentIndex())
        panel.spin_resample.setValue(20.)
        panel._psth_timer.stop()
        view = panel.behavior_zone_panel
        for index in range(view.selectors["behavior"].count()):
            item = view.selectors["behavior"].item(index)
            item.setCheckState(QtCore.Qt.CheckState.Checked if item.text() in demo_labels
                               else QtCore.Qt.CheckState.Unchecked)
        panel.combo_behavior_name.setCurrentText(selected)
        panel._compute_psth()
        view._compute()
        if not view._last_rows or not any(row["events"] for row in view._last_rows):
            raise RuntimeError("The example did not produce complete behavior bouts.")
        print("EXAMPLE_RESULTS", view._last_rows, flush=True)

        if args.zone_workbook:
            zone_names = sorted({name for source in panel._behavior_sources.values()
                                 for name in source.get("behaviors", {}) if name.startswith("Zone: ")})
            if not zone_names:
                raise RuntimeError("The workbook has no zone flags to test.")
            panel._set_event_category("zone")
            panel.combo_behavior_name.setCurrentText(zone_names[0])
            panel._compute_psth()
            if panel._last_mat is None or not np.asarray(panel._last_mat).size:
                raise RuntimeError("The zone example produced no PSTH rows.")
            print("ZONE_EXAMPLE_OK", zone_names, np.asarray(panel._last_mat).shape, flush=True)
            panel._set_event_category("behavior")
            panel.combo_behavior_name.setCurrentText(selected)
            panel._compute_psth()
            view._compute()

        banner = QtWidgets.QLabel("DEMO — " + provenance + " Not an experimental result.")
        banner.setWordWrap(True)
        banner.setStyleSheet("color: #efc778; padding: 6px; font-weight: 600;")
        panel._right_panel.layout().insertWidget(0, banner)
        window._dirty_poll.stop()
        window.setWindowTitle("pyBer — VERIFIED labels — Group — SIMULATED fiber" if args.group else
                              "pyBer — VERIFIED labels — Individual — SIMULATED fiber")
        if args.zone_workbook:
            window.setWindowTitle("pyBer — BEHAVIOR + ZONE TEST — SIMULATED fiber")
        window.top_bar.set_project_name("DEMO — simulated fiber", dirty=False)
        panel.set_current_source_label("DEMO — simulated fiber", "DEMO")
        window.showNormal()
        panel._hide_all_section_popups()
        app.processEvents()
        chosen_item = view.selectors["behavior"].findItems(selected, QtCore.Qt.MatchFlag.MatchExactly)
        if chosen_item:
            view.selectors["behavior"].scrollToItem(chosen_item[0])
        panel.behavior_psth_bar.heat_button.click()
        if args.screenshot:
            window.grab().save(str(args.screenshot))
        if args.smoke:
            panel.tab_visual_mode.setCurrentIndex(1)
            view._compute()
            assert view._last_group_rows and all(row["recordings"] == len(sources)
                                                for row in view._last_group_rows)
            print("GROUP_EXAMPLE_OK", len(sources), "recordings", flush=True)
            panel._project_dirty = False
            panel._psth_timer.stop()
            return 0
        window.raise_()
        window.activateWindow()
        def reveal_psth():
            panel._hide_all_section_popups()
            panel.behavior_psth_bar.heat_button.click()
            view._compute()
            print("EXAMPLE_READY", int(window.winId()), flush=True)
        QtCore.QTimer.singleShot(1500, reveal_psth)
        return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
