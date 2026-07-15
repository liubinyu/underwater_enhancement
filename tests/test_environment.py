from scripts.check_environment import collect_environment


def test_environment_report_has_required_fields() -> None:
    report = collect_environment()
    required = {
        "python",
        "torch",
        "cuda_available",
        "cuda_version",
        "gpu_name",
        "gpu_memory_gb",
        "transformers",
        "ms_swift",
        "peft",
        "trl",
        "bitsandbytes_available",
        "bf16_supported",
    }
    assert required <= report.keys()
    assert isinstance(report["cuda_available"], bool)
    assert isinstance(report["bf16_supported"], bool)
