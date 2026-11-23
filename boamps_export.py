"""BoAmps output adapter for the pinned CodeCarbon 3.3.3 meter."""
from pathlib import Path

from codecarbon.external.logger import logger
from codecarbon.output_methods.boamps import BoAmpsOutput
from codecarbon.output_methods.boamps.mapper import map_emissions_to_boamps

from storage import atomic_json


def tracking_modes(sources):
    """Translate the actual Linux metering backends from the monitoring receipt."""
    cpu_mode = gpu_mode = None
    for source in sources:
        if source["class"] == "CPU":
            cpu_mode = {"intel_rapl": "rapl", "cpu_load": "cpu_load", "constant": "constant"}.get(source["mode"])
        elif source["class"] == "GPU":
            labels = {"NvidiaGPUDevice": "nvml", "AMDGPUDevice": "amdsmi"}
            by_index = {device["index"]: labels.get(device["backend"]) for device in source["devices"]}
            modes = {by_index.get(index) for index in source["monitored_gpu_indices"]}
            # One BoAmps field for the entire GPU contribution. Don't infer a
            # backend when the selected devices are absent, unknown or mixed.
            if len(modes) == 1:
                gpu_mode = modes.pop()
    return cpu_mode, gpu_mode


class TrackingBoAmpsOutput(BoAmpsOutput):
    """Use CodeCarbon's mapper and add provenance before the first write."""

    sources = ()  # Replaced with the actual source receipt after hardware setup.

    def out(self, total, delta):
        try:
            report = map_emissions_to_boamps(
                total, task=self._task, header=self._header, quality=self._quality,
                infra_overrides=self._infra_overrides, environment_overrides=self._environment_overrides,
            )
            cpu, gpu = tracking_modes(self.sources)
            for measure in report.measures:
                measure.cpu_tracking_mode = cpu
                measure.gpu_tracking_mode = gpu
            path = Path(self._output_dir) / f"boamps_report_{total.run_id}.json"
            atomic_json(path, report.to_dict())
            self.save_file_path = str(path)
            logger.info(f"BoAmps report saved to {path.resolve()}")
        except Exception as exc:
            # Preserve the native exporter's failure behavior: CSV can still
            # be written, and the missing BoAmps report will fail validation.
            logger.error(f"Failed to write BoAmps report: {exc}", exc_info=True)
