from pathlib import Path

from model_server.config import ServerConfig
from model_server.doctor import Check, Doctor, DoctorReport


def test_doctor_report_pass_warn_fail_and_blocking():
    report = DoctorReport([Check("PASS", "a", "ok"), Check("WARN", "b", "hmm"), Check("FAIL", "c", "bad", "fix it")], {})
    assert report.blocking
    assert report.as_dict()["summary"] == {"PASS": 1, "WARN": 1, "FAIL": 1}


def test_every_fail_requires_remedy(tmp_path: Path):
    doctor = Doctor(ServerConfig(model="x"), tmp_path)
    try:
        doctor.add("FAIL", "bad", "message")
    except AssertionError:
        pass
    else:
        raise AssertionError("FAIL without remedy was accepted")


def test_bfloat16_rejected_on_compute_capability_70(tmp_path: Path):
    cfg = ServerConfig(model="org/model", dtype="bfloat16", gpus="0")
    doctor = Doctor(cfg, tmp_path)
    doctor._configuration(
        {"gpus": [{"index": 0, "compute_capability": "7.0", "free_mib": 30000, "total_mib": 32000}]},
        {}, {"safetensors": {"total": 1_000_000}},
    )
    check = next(c for c in doctor.checks if c.name == "dtype_compatibility")
    assert check.status == "FAIL"
    assert "Ampere" in check.message
    assert check.remedy


def test_gams3_vllm_is_blocked(tmp_path: Path):
    doctor = Doctor(ServerConfig(backend="vllm", model="cjvt/GaMS3-12B-Instruct", dtype="float32"), tmp_path)
    doctor._configuration({"gpus": []}, {}, {})
    check = next(c for c in doctor.checks if c.name == "backend_model_compatibility")
    assert check.status == "FAIL"
    assert "transformers" in check.remedy.lower()


def test_gams3_known_v100_int8_is_warning_not_false_pass(tmp_path: Path):
    cfg = ServerConfig(model="cjvt/GaMS3-12B-Instruct", quantization="int8", dtype="float32")
    doctor = Doctor(cfg, tmp_path)
    doctor._configuration(
        {"gpus": [{"index": 0, "compute_capability": "7.0", "free_mib": 32000, "total_mib": 32000}]},
        {}, {"safetensors": {"total": 12_000_000_000}},
    )
    check = next(c for c in doctor.checks if c.name == "quantization_compatibility")
    assert check.status == "WARN"
    assert "user-verified" in check.message

